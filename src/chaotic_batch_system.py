"""
策略1：乱序竞争批处理系统

架构：
Reader0 ──┐                                                   ┌─ VideoProcessor0 (heap+tracker) → Saver0
Reader1 ──┼→ shared_queue → ChaoticBatchProcessor → dispatcher ┼─ VideoProcessor1 (heap+tracker) → Saver1
Reader2 ──┘   (竞争写入)     (按batch_size打包推理)              └─ VideoProcessor2 (heap+tracker) → Saver2

核心思路：
- 多个 Reader 线程竞争写入同一个 shared_queue，帧自然乱序
- ChaoticBatchProcessor 按 batch_size 从 shared_queue 取帧打包推理
- ReorderDispatcher 按 video_id 分发到各 VideoProcessor
- 每个 VideoProcessor 使用小顶堆按 frame_id 重排序，连续弹出后送入 tracker
"""

import heapq
import logging
import threading
import time
from queue import Queue, Empty
from typing import Callable, Dict, List, Optional
from dataclasses import dataclass, field

from pipeline_data import FrameData
from pipeline_modules import Saver
from video_source import VideoSource
from performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)


class ChaoticReader:
    """竞争写入器 — 读取视频帧写入共享队列"""

    def __init__(self, video_source: VideoSource, shared_queue: Queue,
                 pipeline_id: str, stop_event: threading.Event):
        self.video_source = video_source
        self.shared_queue = shared_queue
        self.pipeline_id = pipeline_id
        self.stop_event = stop_event
        self.stats = {'total_frames': 0, 'dropped_frames': 0}

    def run(self):
        logger.info(f"[{self.pipeline_id}] ChaoticReader started")

        if not self.video_source.open():
            logger.error(f"[{self.pipeline_id}] Failed to open video source")
            return

        frame_id = 0
        try:
            while not self.stop_event.is_set():
                ret, frame = self.video_source.read()
                if not ret or frame is None:
                    logger.info(f"[{self.pipeline_id}] Reached end of video")
                    break

                frame_id += 1
                frame_data = FrameData(
                    frame_id=frame_id,
                    timestamp=time.time(),
                    frame=frame,
                    video_id=self.pipeline_id,
                    video_name=self.video_source.name,
                )
                PerformanceMonitor.probe(self.pipeline_id, frame_id, "start")

                try:
                    self.shared_queue.put(frame_data, timeout=5.0)
                    self.stats['total_frames'] += 1
                except Exception:
                    self.stats['dropped_frames'] += 1

        except Exception as e:
            logger.error(f"[{self.pipeline_id}] ChaoticReader error: {e}")
        finally:
            self.video_source.close()
            try:
                self.shared_queue.put(None, timeout=1.0)
            except Exception:
                pass
            logger.info(f"[{self.pipeline_id}] ChaoticReader stopped. stats={self.stats}")


class ChaoticBatchProcessor:
    """乱序批推理器 — 从共享队列按 batch_size 打包推理"""

    def __init__(self, shared_queue: Queue, dispatch_queue: Queue,
                 inference_func: Callable, batch_size: int,
                 num_videos: int, stop_event: threading.Event):
        self.shared_queue = shared_queue
        self.dispatch_queue = dispatch_queue
        self.inference_func = inference_func
        self.batch_size = batch_size
        self.num_videos = num_videos
        self.stop_event = stop_event
        self.stats = {'total_batches': 0, 'total_frames': 0, 'inference_time': 0.0}

    def run(self):
        logger.info("ChaoticBatchProcessor started")
        finished_readers = 0

        try:
            while not self.stop_event.is_set():
                batch_frames = []
                batch_metas: List[FrameData] = []

                # 收集一个 batch
                while len(batch_frames) < self.batch_size:
                    try:
                        frame_data = self.shared_queue.get(timeout=0.5)
                    except Empty:
                        if self.stop_event.is_set():
                            break
                        # 队列暂时为空但还没结束，用已收集的帧先推理
                        if batch_frames:
                            break
                        continue

                    if frame_data is None:
                        finished_readers += 1
                        if finished_readers >= self.num_videos:
                            logger.info("All readers finished")
                            break
                        continue

                    batch_frames.append(frame_data.frame)
                    batch_metas.append(frame_data)

                if not batch_frames:
                    if finished_readers >= self.num_videos:
                        break
                    continue

                # 批推理
                start_time = time.time()
                try:
                    detections_list = self.inference_func(batch_frames)
                except Exception as e:
                    logger.error(f"Batch inference error: {e}")
                    detections_list = [[] for _ in batch_frames]
                inference_time = time.time() - start_time

                # 将推理结果附着到 FrameData 并送入 dispatch_queue
                for frame_data, detections in zip(batch_metas, detections_list):
                    frame_data.detections = detections
                    try:
                        self.dispatch_queue.put(frame_data, timeout=2.0)
                    except Exception as e:
                        logger.warning(f"Failed to dispatch: {e}")

                self.stats['total_batches'] += 1
                self.stats['total_frames'] += len(batch_frames)
                self.stats['inference_time'] += inference_time

        except Exception as e:
            logger.error(f"ChaoticBatchProcessor error: {e}")
        finally:
            # 发送结束信号
            try:
                self.dispatch_queue.put(None, timeout=1.0)
            except Exception:
                pass
            logger.info(f"ChaoticBatchProcessor stopped. stats={self.stats}")


@dataclass(order=True)
class HeapItem:
    """小顶堆元素，按 frame_id 排序"""
    frame_id: int
    frame_data: FrameData = field(compare=False)


class VideoProcessor:
    """每视频处理器 — 小顶堆重排序 + 追踪"""

    def __init__(self, video_id: str, tracker_instance,
                 output_queue: Queue, stop_event: threading.Event):
        self.video_id = video_id
        self.tracker = tracker_instance
        self.output_queue = output_queue
        self.stop_event = stop_event

        self.counter = 1  # 期望的下一帧序号
        self.heap: List[HeapItem] = []
        self.lock = threading.Lock()
        self.new_frame_event = threading.Event()
        self.finished = False
        self.stats = {'total_frames': 0, 'reordered_frames': 0}

    def push_frame(self, frame_data: FrameData):
        """外部调用：将帧压入小顶堆"""
        with self.lock:
            heapq.heappush(self.heap, HeapItem(frame_data.frame_id, frame_data))
        self.new_frame_event.set()

    def signal_finished(self):
        """外部调用：通知不再有新帧"""
        self.finished = True
        self.new_frame_event.set()

    def run(self):
        """处理循环：从堆中连续弹出并追踪"""
        logger.info(f"[{self.video_id}] VideoProcessor started")

        try:
            while not self.stop_event.is_set():
                self.new_frame_event.wait(timeout=1.0)
                self.new_frame_event.clear()

                self._flush_consecutive()
                print("calling old method")

                if self.finished:
                    # 再 flush 一次，确保堆中剩余帧被处理
                    self._flush_consecutive()
                    break

        except Exception as e:
            logger.error(f"[{self.video_id}] VideoProcessor error: {e}")
        finally:
            # 强制清空堆中剩余帧（可能有间隔的帧）
            self._flush_all_remaining()
            try:
                self.output_queue.put(None, timeout=1.0)
            except Exception:
                pass
            logger.info(f"[{self.video_id}] VideoProcessor stopped. stats={self.stats}")

    def _flush_consecutive(self):
        """连续弹出堆顶 frame_id == counter 的帧"""
        with self.lock:
            while self.heap and self.heap[0].frame_id == self.counter:
                item = heapq.heappop(self.heap)
                self._process_frame(item.frame_data)
                self.counter += 1

    def _flush_all_remaining(self):
        """结束时按序弹出堆中所有剩余帧"""
        with self.lock:
            while self.heap:
                item = heapq.heappop(self.heap)
                self._process_frame(item.frame_data)

    def _process_frame(self, frame_data: FrameData):
        """追踪并送入输出队列"""
        if frame_data.frame is not None and self.tracker is not None:
            track_results = self.tracker.update(frame_data.frame, frame_data.detections)
            if track_results is not None:
                frame_data.detections = track_results

        self.stats['total_frames'] += 1

        try:
            self.output_queue.put(frame_data, timeout=5.0)
        except Exception as e:
            logger.warning(f"[{self.video_id}] Failed to output frame: {e}")


class ReorderDispatcher:
    """乱序恢复分发器 — 按 video_id 分发到各 VideoProcessor"""

    def __init__(self, dispatch_queue: Queue,
                 video_processors: Dict[str, VideoProcessor],
                 stop_event: threading.Event):
        self.dispatch_queue = dispatch_queue
        self.video_processors = video_processors
        self.stop_event = stop_event

    def run(self):
        logger.info("ReorderDispatcher started")

        try:
            while not self.stop_event.is_set():
                try:
                    frame_data = self.dispatch_queue.get(timeout=1.0)
                except Empty:
                    continue

                if frame_data is None:
                    logger.info("ReorderDispatcher received end signal")
                    break

                video_id = frame_data.video_id
                processor = self.video_processors.get(video_id)
                if processor is not None:
                    processor.push_frame(frame_data)
                else:
                    logger.warning(f"Unknown video_id: {video_id}")

        except Exception as e:
            logger.error(f"ReorderDispatcher error: {e}")
        finally:
            for processor in self.video_processors.values():
                processor.signal_finished()
            logger.info("ReorderDispatcher stopped")


class ChaoticBatchPipeline:
    """策略1顶层管理器"""

    def __init__(self,
                 video_sources: List,
                 inference_func: Callable,
                 tracker_factory: Callable,
                 save_func: Callable,
                 output_dir: str = "result",
                 batch_size: int = 32,
                 queue_size: int = 200):
        self.video_sources = video_sources
        self.inference_func = inference_func
        self.tracker_factory = tracker_factory
        self.save_func = save_func
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.num_videos = len(video_sources)

        self.stop_event = threading.Event()
        self.threads: List[threading.Thread] = []

        # 共享队列
        self.shared_queue = Queue(maxsize=queue_size)
        self.dispatch_queue = Queue(maxsize=queue_size)

        # Readers
        self.readers: List[ChaoticReader] = []
        for i, source in enumerate(video_sources):
            pipeline_id = f"video_{i}"
            self.readers.append(ChaoticReader(
                video_source=source,
                shared_queue=self.shared_queue,
                pipeline_id=pipeline_id,
                stop_event=self.stop_event,
            ))

        # Batch processor
        self.batch_processor = ChaoticBatchProcessor(
            shared_queue=self.shared_queue,
            dispatch_queue=self.dispatch_queue,
            inference_func=inference_func,
            batch_size=batch_size,
            num_videos=self.num_videos,
            stop_event=self.stop_event,
        )

        # VideoProcessors + Savers
        self.video_processors: Dict[str, VideoProcessor] = {}
        self.savers: List[Saver] = []
        self.save_queues: List[Queue] = []

        for i in range(self.num_videos):
            pipeline_id = f"video_{i}"
            tracker_instance = self.tracker_factory(pipeline_id)
            save_queue = Queue(maxsize=queue_size)
            self.save_queues.append(save_queue)

            vp = VideoProcessor(
                video_id=pipeline_id,
                tracker_instance=tracker_instance,
                output_queue=save_queue,
                stop_event=self.stop_event,
            )
            self.video_processors[pipeline_id] = vp

            saver = Saver(
                input_queue=save_queue,
                output_dir=output_dir,
                save_func=save_func,
                pipeline_id=pipeline_id,
                stop_event=self.stop_event,
            )
            self.savers.append(saver)

        # Dispatcher
        self.dispatcher = ReorderDispatcher(
            dispatch_queue=self.dispatch_queue,
            video_processors=self.video_processors,
            stop_event=self.stop_event,
        )

        logger.info(f"ChaoticBatchPipeline: {self.num_videos} videos, batch_size={batch_size}")

    def start(self):
        logger.info("Starting ChaoticBatchPipeline...")

        # Readers
        for i, reader in enumerate(self.readers):
            t = threading.Thread(target=reader.run, name=f"ChaoticReader_{i}", daemon=True)
            t.start()
            self.threads.append(t)

        # Batch processor
        t = threading.Thread(target=self.batch_processor.run,
                             name="ChaoticBatchProcessor", daemon=True)
        t.start()
        self.threads.append(t)

        # Dispatcher
        t = threading.Thread(target=self.dispatcher.run,
                             name="ReorderDispatcher", daemon=True)
        t.start()
        self.threads.append(t)

        # VideoProcessors
        for vid, vp in self.video_processors.items():
            t = threading.Thread(target=vp.run, name=f"VideoProcessor_{vid}", daemon=True)
            t.start()
            self.threads.append(t)

        # Savers
        for i, saver in enumerate(self.savers):
            t = threading.Thread(target=saver.run, name=f"Saver_{i}", daemon=True)
            t.start()
            self.threads.append(t)

        logger.info(f"Started {len(self.threads)} threads")

    def stop(self):
        logger.info("Stopping ChaoticBatchPipeline...")
        self.stop_event.set()
        for q in [self.shared_queue, self.dispatch_queue] + self.save_queues:
            try:
                while not q.empty():
                    q.get_nowait()
            except Exception:
                pass

    def wait(self, timeout: Optional[float] = None) -> bool:
        logger.info("Waiting for all threads...")
        all_done = True
        for t in self.threads:
            t.join(timeout=timeout if timeout else 5.0)
            if t.is_alive():
                all_done = False
                logger.warning(f"Thread {t.name} still alive")
        return all_done

    def get_stats(self) -> Dict:
        return {
            'num_videos': self.num_videos,
            'batch_size': self.batch_size,
            'batch_processor': self.batch_processor.stats.copy(),
            'readers': [r.stats.copy() for r in self.readers],
            'video_processors': {vid: vp.stats.copy()
                                 for vid, vp in self.video_processors.items()},
            'savers': [s.stats.copy() for s in self.savers],
        }
