"""
多视频批处理推理系统

架构：
Reader0 ──┐                              ┌──> Tracker0 ──> Saver0
Reader1 ──┼──> BatchProcessor ──────────┼──> Tracker1 ──> Saver1
Reader2 ──┤   (GPU批推理)                ├──> Tracker2 ──> Saver2
...      ─┘                              └──> ...

k值计算：
- k = batch_size // n（向下取整）
- 最后一个视频补齐
"""

import threading
import logging
import time
from queue import Queue, Empty
from typing import List, Optional, Callable, Dict, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def calculate_k_values(num_videos: int, batch_size: int = 32) -> List[int]:
    """
    计算每个视频的k值（可以打进批次的帧数量）

    Args:
        num_videos: 视频数量
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

    k_values = [base_k] * (num_videos - 1)
    last_k = base_k + remainder
    k_values.append(last_k)

    logger.info(f"k_values for {num_videos} videos (batch={batch_size}): {k_values}, sum={sum(k_values)}")
    return k_values


@dataclass
class FrameMeta:
    """帧元信息"""
    video_idx: int
    frame_idx: int
    frame_data: Any


class BatchCollector:
    """批次收集器 - 从各视频队列按k值收集帧"""

    def __init__(self, num_videos: int, batch_size: int = 32):
        self.num_videos = num_videos
        self.batch_size = batch_size
        self.k_values = calculate_k_values(num_videos, batch_size)

        logger.info(f"BatchCollector initialized: {num_videos} videos, batch_size={batch_size}")

    def collect_batch(self, queues: List[Queue],
                      video_finished: List[bool],
                      timeout: float = 0.1) -> Tuple[List[Any], List[FrameMeta], List[int]]:
        """从各视频队列按k值收集帧"""
        batch_frames = []
        frame_metas = []
        actual_k_values = []

        for video_idx, target_k in enumerate(self.k_values):
            if video_finished[video_idx]:
                actual_k_values.append(0)
                continue

            frames_collected = 0

            while frames_collected < target_k:
                try:
                    frame_data = queues[video_idx].get(timeout=timeout)

                    if frame_data is None:
                        video_finished[video_idx] = True
                        logger.info(f"Video {video_idx} finished")
                        break

                    meta = FrameMeta(
                        video_idx=video_idx,
                        frame_idx=frame_data.frame_id,
                        frame_data=frame_data
                    )

                    batch_frames.append(frame_data.frame)
                    frame_metas.append(meta)
                    frames_collected += 1

                except Empty:
                    break

            actual_k_values.append(frames_collected)

        return batch_frames, frame_metas, actual_k_values


class ResultDistributor:
    """结果分发器 - 使用head/offset机制分发推理结果"""

    def __init__(self, num_videos: int):
        self.num_videos = num_videos

    def distribute(self, frame_metas: List[FrameMeta],
                   detections: List[Any],
                   actual_k_values: List[int]) -> Dict[int, List[Any]]:
        """将推理结果按视频分发"""
        result_dict: Dict[int, List[Any]] = {i: [] for i in range(self.num_videos)}

        head = 0
        for video_idx, offset in enumerate(actual_k_values):
            if offset == 0:
                continue

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
    """多视频批处理器 - 收集 -> 推理 -> 分发"""

    def __init__(self,
                 input_queues: List[Queue],
                 output_queues: List[Queue],
                 inference_func: Callable,
                 batch_size: int = 32,
                 stop_event: Optional[threading.Event] = None):
        self.input_queues = input_queues
        self.output_queues = output_queues
        self.inference_func = inference_func
        self.batch_size = batch_size
        self.stop_event = stop_event or threading.Event()

        self.num_videos = len(input_queues)
        self.collector = BatchCollector(self.num_videos, batch_size)
        self.distributor = ResultDistributor(self.num_videos)

        self.video_finished = [False] * self.num_videos

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
                if all(self.video_finished):
                    logger.info("All videos finished")
                    break

                batch_frames, frame_metas, actual_k_values = self.collector.collect_batch(
                    self.input_queues,
                    self.video_finished,
                    timeout=0.5
                )

                if not batch_frames:
                    time.sleep(0.01)
                    continue

                start_time = time.time()
                try:
                    detections = self.inference_func(batch_frames)
                except Exception as e:
                    logger.error(f"Batch inference error: {e}")
                    detections = [[] for _ in batch_frames]

                inference_time = time.time() - start_time

                result_dict = self.distributor.distribute(
                    frame_metas, detections, actual_k_values
                )

                for video_idx, frames in result_dict.items():
                    for frame_data in frames:
                        try:
                            self.output_queues[video_idx].put(frame_data, timeout=2.0)
                        except Exception as e:
                            logger.warning(f"Failed to put to queue {video_idx}: {e}")

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
            for i, q in enumerate(self.output_queues):
                try:
                    q.put(None, timeout=1.0)
                except:
                    pass

            logger.info(f"MultiVideoBatchProcessor stopped. Stats: {self.stats}")

    def get_stats(self) -> Dict:
        return self.stats.copy()


class MultiVideoPipeline:
    """多视频批处理流水线"""

    def __init__(self,
                 video_sources: List,
                 inference_func: Callable,
                 tracker_factory: Callable,
                 save_func: Callable,
                 output_dir: str = "result",
                 batch_size: int = 32,
                 queue_size: int = 100):
        self.video_sources = video_sources
        self.inference_func = inference_func
        self.tracker_factory = tracker_factory
        self.save_func = save_func
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.queue_size = queue_size
        self.num_videos = len(video_sources)

        self.stop_event = threading.Event()

        self.read_queues: List[Queue] = []
        self.infer_queues: List[Queue] = []
        self.save_queues: List[Queue] = []

        self.readers = []
        self.batch_processor: Optional[MultiVideoBatchProcessor] = None
        self.trackers = []
        self.savers = []

        self.threads: List[threading.Thread] = []

        self._setup_pipeline()

        logger.info(f"MultiVideoPipeline: {self.num_videos} videos, batch_size={batch_size}")

    def _setup_pipeline(self):
        """设置流水线组件"""
        from pipeline_modules import Reader, Tracker, Saver

        logger.info("Initializing trackers...")
        tracker_instances = []
        for i in range(self.num_videos):
            pipeline_id = f"video_{i}"
            tracker_instance = self.tracker_factory(pipeline_id)
            tracker_instances.append(tracker_instance)
        logger.info(f"All {self.num_videos} trackers initialized")

        for i, video_source in enumerate(self.video_sources):
            pipeline_id = f"video_{i}"

            read_queue = Queue(maxsize=self.queue_size)
            infer_queue = Queue(maxsize=self.queue_size)
            save_queue = Queue(maxsize=self.queue_size)

            self.read_queues.append(read_queue)
            self.infer_queues.append(infer_queue)
            self.save_queues.append(save_queue)

            reader = Reader(
                video_source=video_source,
                output_queue=read_queue,
                pipeline_id=pipeline_id,
                stop_event=self.stop_event
            )
            self.readers.append(reader)

            tracker = Tracker(
                input_queue=infer_queue,
                output_queue=save_queue,
                tracker_instance=tracker_instances[i],
                pipeline_id=pipeline_id,
                stop_event=self.stop_event
            )
            self.trackers.append(tracker)

            saver = Saver(
                input_queue=save_queue,
                output_dir=self.output_dir,
                save_func=self.save_func,
                pipeline_id=pipeline_id,
                stop_event=self.stop_event
            )
            self.savers.append(saver)

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

        for i, reader in enumerate(self.readers):
            t = threading.Thread(target=reader.run, name=f"Reader_{i}", daemon=True)
            t.start()
            self.threads.append(t)

        t = threading.Thread(target=self.batch_processor.run,
                           name="BatchProcessor", daemon=True)
        t.start()
        self.threads.append(t)

        for i, tracker in enumerate(self.trackers):
            t = threading.Thread(target=tracker.run, name=f"Tracker_{i}", daemon=True)
            t.start()
            self.threads.append(t)

        for i, saver in enumerate(self.savers):
            t = threading.Thread(target=saver.run, name=f"Saver_{i}", daemon=True)
            t.start()
            self.threads.append(t)

        logger.info(f"Started {len(self.threads)} threads")

    def stop(self):
        """停止所有线程"""
        logger.info("Stopping MultiVideoPipeline...")
        self.stop_event.set()

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
