"""
多视频批处理推理系统 v2

术语：
=====
- cv: captured video (视频源)
- k: 每个视频可以打进批次的帧数量，k = batch_size // n（向下取整），最后一个补齐
- offset: 等于k，从buffer取帧时的偏移量
- head: 读指针，指向当前读取位置

k值计算示例：
============
若 n=15, batch_size=32:
  基础k = 32 // 15 = 2
  k_values = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,4]  # 最后一个补齐

流程：
=====
1. 各视频读取到各自队列: queue[i] = [frames in cv_i]

2. 根据batch_size计算每个视频的k值，打包进batch:
   batch = [cv0f0, cv0f1, ..., cv0f(k0-1), cv1f0, cv1f1, ..., cv1f(k1-1), ...]
   同时保存帧元信息做区分

3. GPU批推理:
   buffer = [cv0f0_det, cv0f1_det, ..., cvnfk_det]

4. 分发到各视频（使用head和offset）:
   head = 0
   for video_idx in range(n):
       offset = k_values[video_idx]
       video_part = buffer[head : head + offset]
       result_dict[video_idx].extend(video_part)
       head += offset

5. 每个视频有独立的YOLO tracker进行追踪，避免竞态条件

架构图：
=======
Reader0 ──┐                              ┌──> Tracker0 (独立YOLO) ──> Saver0
Reader1 ──┼──> BatchProcessor ──────────┼──> Tracker1 (独立YOLO) ──> Saver1
Reader2 ──┤   (GPU批推理)                ├──> Tracker2 (独立YOLO) ──> Saver2
...      ─┘                              └──> ...
"""

import threading
import logging
import time
from queue import Queue, Empty
from typing import List, Optional, Callable, Dict, Any, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


def calculate_k_values(num_videos: int, batch_size: int = 32) -> List[int]:
    """
    计算每个视频的k值（可以打进批次的帧数量）
    
    规则：
    - 基础k = batch_size // num_videos（向下取整）
    - 最后一个视频补齐剩余帧数
    
    例如 n=15, batch_size=32:
    - 基础k = 32 // 15 = 2
    - 前14个视频: k=2 (共28帧)
    - 最后1个视频: k=4 (补齐到32帧)
    - k_values = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,4]
    
    Args:
        num_videos: 视频数量 n
        batch_size: 批处理大小
    
    Returns:
        每个视频的k值列表
    """
    if num_videos == 0:
        return []
    
    if num_videos == 1:
        return [batch_size]
    
    base_k = batch_size // num_videos
    remainder = batch_size % num_videos
    
    # 前 n-1 个视频用基础k值
    k_values = [base_k] * (num_videos - 1)
    
    # 最后一个视频补齐
    last_k = base_k + remainder
    k_values.append(last_k)
    
    logger.info(f"k_values for {num_videos} videos (batch={batch_size}): {k_values}, sum={sum(k_values)}")
    return k_values


@dataclass
class FrameMeta:
    """帧元信息 - 用于追踪帧的来源"""
    video_idx: int          # 视频索引
    frame_idx: int          # 帧在视频中的序号
    frame_data: Any         # 原始FrameData对象


class BatchCollector:
    """
    批次收集器
    
    从各视频队列按k值收集帧，打包成batch
    """
    
    def __init__(self, num_videos: int, batch_size: int = 32):
        self.num_videos = num_videos
        self.batch_size = batch_size
        self.k_values = calculate_k_values(num_videos, batch_size)
        
        logger.info(f"BatchCollector initialized: {num_videos} videos, batch_size={batch_size}")
    
    def collect_batch(self, queues: List[Queue], 
                      video_finished: List[bool],
                      timeout: float = 0.1) -> Tuple[List[Any], List[FrameMeta], List[int]]:
        """
        从各视频队列按k值收集帧
        
        收集顺序：
        [cv0f0, cv0f1, ..., cv0f(k0-1), cv1f0, cv1f1, ..., cv1f(k1-1), ...]
        
        Args:
            queues: 各视频的帧队列
            video_finished: 各视频是否已结束
            timeout: 等待超时（每帧）
        
        Returns:
            (batch_frames, frame_metas, actual_k_values)
        """
        batch_frames = []
        frame_metas = []
        actual_k_values = []  # 实际收集到的k值
        
        for video_idx, target_k in enumerate(self.k_values):
            if video_finished[video_idx]:
                actual_k_values.append(0)
                continue
            
            frames_collected = 0
            
            # 从该视频连续取target_k帧
            while frames_collected < target_k:
                try:
                    frame_data = queues[video_idx].get(timeout=timeout)
                    
                    if frame_data is None:
                        # 视频结束信号
                        video_finished[video_idx] = True
                        logger.info(f"Video {video_idx} finished")
                        break
                    
                    # 记录元信息
                    meta = FrameMeta(
                        video_idx=video_idx,
                        frame_idx=frame_data.frame_id,
                        frame_data=frame_data
                    )
                    
                    batch_frames.append(frame_data.frame)
                    frame_metas.append(meta)
                    frames_collected += 1
                    
                except Empty:
                    # 队列暂时为空，跳过继续下一个视频
                    break
            
            actual_k_values.append(frames_collected)
        
        return batch_frames, frame_metas, actual_k_values


class ResultDistributor:
    """
    结果分发器
    
    使用head和offset机制将推理结果分发到各视频
    """
    
    def __init__(self, num_videos: int):
        self.num_videos = num_videos
    
    def distribute(self, frame_metas: List[FrameMeta], 
                   detections: List[Any],
                   actual_k_values: List[int]) -> Dict[int, List[Any]]:
        """
        将推理结果按视频分发
        
        使用head和offset机制：
        head = 0
        for video_idx in range(n):
            offset = actual_k_values[video_idx]
            video_frames = buffer[head : head + offset]
            head += offset
        
        Args:
            frame_metas: 帧元信息列表
            detections: 推理结果列表
            actual_k_values: 每个视频实际收集的帧数
        
        Returns:
            dict[video_idx] -> list[FrameData with detection]
        """
        result_dict: Dict[int, List[Any]] = {i: [] for i in range(self.num_videos)}
        
        head = 0
        for video_idx, offset in enumerate(actual_k_values):
            if offset == 0:
                continue
            
            # 取出 [head, head + offset) 范围的帧
            video_metas = frame_metas[head : head + offset]
            video_detections = detections[head : head + offset]
            
            for meta, detection in zip(video_metas, video_detections):
                frame_data = meta.frame_data
                frame_data.detections = detection
                frame_data.video_index = video_idx
                result_dict[video_idx].append(frame_data)
            
            head += offset
        
        return result_dict


class MultiVideoBatchProcessor:
    """
    多视频批处理器
    
    整合：收集 -> 推理 -> 分发
    """
    
    def __init__(self, 
                 input_queues: List[Queue],
                 output_queues: List[Queue],
                 inference_func: Callable,
                 batch_size: int = 32,
                 stop_event: Optional[threading.Event] = None):
        """
        Args:
            input_queues: 各视频的输入队列（来自Reader）
            output_queues: 各视频的输出队列（到Tracker）
            inference_func: 批推理函数 (List[frames]) -> List[detections]
            batch_size: 批处理大小
            stop_event: 停止信号
        """
        self.input_queues = input_queues
        self.output_queues = output_queues
        self.inference_func = inference_func
        self.batch_size = batch_size
        self.stop_event = stop_event or threading.Event()
        
        self.num_videos = len(input_queues)
        self.collector = BatchCollector(self.num_videos, batch_size)
        self.distributor = ResultDistributor(self.num_videos)
        
        # 视频结束标记
        self.video_finished = [False] * self.num_videos
        
        # 统计
        self.stats = {
            'total_batches': 0,
            'total_frames': 0,
            'inference_time': 0.0,
        }
        
        logger.info(f"MultiVideoBatchProcessor: {self.num_videos} videos, "
                   f"batch_size={batch_size}, k_values={self.collector.k_values}")
    
    def run(self):
        """执行批处理循环"""
        logger.info("MultiVideoBatchProcessor started")
        
        try:
            while not self.stop_event.is_set():
                # 检查是否所有视频都已结束
                if all(self.video_finished):
                    logger.info("All videos finished")
                    break
                
                # 1. 收集batch（按k值从各视频取帧）
                batch_frames, frame_metas, actual_k_values = self.collector.collect_batch(
                    self.input_queues, 
                    self.video_finished,
                    timeout=0.5
                )
                
                if not batch_frames:
                    time.sleep(0.01)
                    continue
                
                # 2. GPU批推理
                start_time = time.time()
                try:
                    detections = self.inference_func(batch_frames)
                except Exception as e:
                    logger.error(f"Batch inference error: {e}")
                    detections = [[] for _ in batch_frames]
                
                inference_time = time.time() - start_time
                
                # 3. 分发结果（使用head/offset机制）
                result_dict = self.distributor.distribute(
                    frame_metas, detections, actual_k_values
                )
                
                # 4. 放入各视频的输出队列
                for video_idx, frames in result_dict.items():
                    for frame_data in frames:
                        try:
                            self.output_queues[video_idx].put(frame_data, timeout=2.0)
                        except Exception as e:
                            logger.warning(f"Failed to put to queue {video_idx}: {e}")
                
                # 更新统计
                self.stats['total_batches'] += 1
                self.stats['total_frames'] += len(batch_frames)
                self.stats['inference_time'] += inference_time
                
                if self.stats['total_batches'] % 10 == 0:
                    logger.debug(f"Batch {self.stats['total_batches']}: "
                               f"{len(batch_frames)} frames, "
                               f"k_actual={actual_k_values}, "
                               f"time={inference_time:.3f}s")
        
        except Exception as e:
            logger.error(f"MultiVideoBatchProcessor error: {e}")
        
        finally:
            # 发送结束信号到所有输出队列
            for i, q in enumerate(self.output_queues):
                try:
                    q.put(None, timeout=1.0)
                except:
                    pass
            
            logger.info(f"MultiVideoBatchProcessor stopped. Stats: {self.stats}")
    
    def get_stats(self) -> Dict:
        return self.stats.copy()


class MultiVideoPipeline:
    """
    多视频批处理流水线
    
    架构：
    Reader0 ──┐                              ┌──> Tracker0 (独立YOLO) ──> Saver0
    Reader1 ──┼──> BatchProcessor ──────────┼──> Tracker1 (独立YOLO) ──> Saver1
    Reader2 ──┤   (GPU批推理)                ├──> Tracker2 (独立YOLO) ──> Saver2
    ...      ─┘                              └──> ...
    
    每个Tracker有独立的YOLO模型，避免竞态条件
    """
    
    def __init__(self,
                 video_sources: List,
                 inference_func: Callable,
                 tracker_factory: Callable,
                 save_func: Callable,
                 output_dir: str = "result",
                 batch_size: int = 32,
                 queue_size: int = 100):
        """
        Args:
            video_sources: 视频源列表
            inference_func: 批推理函数
            tracker_factory: 追踪器工厂函数(pipeline_id) -> tracker，每个视频独立YOLO
            save_func: 保存函数
            output_dir: 输出目录
            batch_size: 批处理大小
            queue_size: 队列大小
        """
        self.video_sources = video_sources
        self.inference_func = inference_func
        self.tracker_factory = tracker_factory
        self.save_func = save_func
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.queue_size = queue_size
        self.num_videos = len(video_sources)
        
        # 停止事件
        self.stop_event = threading.Event()
        
        # 队列
        self.read_queues: List[Queue] = []      # Reader -> BatchProcessor
        self.infer_queues: List[Queue] = []     # BatchProcessor -> Tracker
        self.save_queues: List[Queue] = []      # Tracker -> Saver
        
        # 组件
        self.readers = []
        self.batch_processor: Optional[MultiVideoBatchProcessor] = None
        self.trackers = []
        self.savers = []
        
        # 线程
        self.threads: List[threading.Thread] = []
        
        self._setup_pipeline()
        
        logger.info(f"MultiVideoPipeline: {self.num_videos} videos, batch_size={batch_size}")
    
    def _setup_pipeline(self):
        """设置流水线组件"""
        from pipeline_modules import Reader, Tracker, Saver
        
        # 先串行创建所有 tracker
        logger.info("Initializing trackers...")
        tracker_instances = []
        for i in range(self.num_videos):
            pipeline_id = f"video_{i}"
            tracker_instance = self.tracker_factory(pipeline_id)
            tracker_instances.append(tracker_instance)
        logger.info(f"All {self.num_videos} trackers initialized")
        
        # 然后创建其他组件
        for i, video_source in enumerate(self.video_sources):
            pipeline_id = f"video_{i}"
            
            # 创建队列
            read_queue = Queue(maxsize=self.queue_size)
            infer_queue = Queue(maxsize=self.queue_size)
            save_queue = Queue(maxsize=self.queue_size)
            
            self.read_queues.append(read_queue)
            self.infer_queues.append(infer_queue)
            self.save_queues.append(save_queue)
            
            # Reader
            reader = Reader(
                video_source=video_source,
                output_queue=read_queue,
                pipeline_id=pipeline_id,
                stop_event=self.stop_event
            )
            self.readers.append(reader)
            
            # Tracker（使用预先创建的 tracker 实例）
            tracker = Tracker(
                input_queue=infer_queue,
                output_queue=save_queue,
                tracker_instance=tracker_instances[i],
                pipeline_id=pipeline_id,
                stop_event=self.stop_event
            )
            self.trackers.append(tracker)
            
            # Saver
            saver = Saver(
                input_queue=save_queue,
                output_dir=self.output_dir,
                save_func=self.save_func,
                pipeline_id=pipeline_id,
                stop_event=self.stop_event
            )
            self.savers.append(saver)
        
        # BatchProcessor
        self.batch_processor = MultiVideoBatchProcessor(
            input_queues=self.read_queues,
            output_queues=self.infer_queues,
            inference_func=self.inference_func,
            batch_size=self.batch_size,
            stop_event=self.stop_event
        )
    
    def start(self):
        """启动所有线程"""
        logger.info("Starting MultiVideoPipeline...")
        
        # 启动Readers
        for i, reader in enumerate(self.readers):
            t = threading.Thread(target=reader.run, name=f"Reader_{i}", daemon=True)
            t.start()
            self.threads.append(t)
        
        # 启动BatchProcessor
        t = threading.Thread(target=self.batch_processor.run, 
                           name="BatchProcessor", daemon=True)
        t.start()
        self.threads.append(t)
        
        # 启动Trackers
        for i, tracker in enumerate(self.trackers):
            t = threading.Thread(target=tracker.run, name=f"Tracker_{i}", daemon=True)
            t.start()
            self.threads.append(t)
        
        # 启动Savers
        for i, saver in enumerate(self.savers):
            t = threading.Thread(target=saver.run, name=f"Saver_{i}", daemon=True)
            t.start()
            self.threads.append(t)
        
        logger.info(f"Started {len(self.threads)} threads")
    
    def stop(self):
        """停止所有线程"""
        logger.info("Stopping MultiVideoPipeline...")
        self.stop_event.set()
        
        # 清空所有队列，防止线程阻塞在队列操作上
        for q in self.read_queues + self.infer_queues + self.save_queues:
            try:
                while not q.empty():
                    q.get_nowait()
            except:
                pass
    
    def wait(self, timeout: Optional[float] = None) -> bool:
        """等待所有线程完成"""
        logger.info("Waiting for all threads...")
        
        all_done = True
        for t in self.threads:
            t.join(timeout=timeout if timeout else 5.0)
            if t.is_alive():
                all_done = False
                logger.warning(f"Thread {t.name} still alive")
        
        return all_done
        
        return all_done
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'num_videos': self.num_videos,
            'batch_size': self.batch_size,
            'k_values': self.batch_processor.collector.k_values if self.batch_processor else [],
            'batch_processor': self.batch_processor.get_stats() if self.batch_processor else {},
            'readers': [r.stats for r in self.readers],
            'trackers': [t.stats for t in self.trackers],
            'savers': [s.stats for s in self.savers],
        }


# 保留旧的类名作为别名，兼容现有代码
MultiVideoBatchPipeline = MultiVideoPipeline
BatchInferencer = MultiVideoBatchProcessor
