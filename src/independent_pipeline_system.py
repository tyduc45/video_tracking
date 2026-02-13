"""
策略2：独立流水线系统

每个视频源维护完整的 Reader → Inferencer → Tracker → Saver 流水线，
各条流水线多线程并行运行，互不干扰。

线程结构（每个视频）：
Reader → read_queue → Inferencer → infer_queue → Tracker → track_queue → Saver
"""

import logging
import threading
import time
from queue import Queue, Empty
from typing import Callable, Dict, List, Optional

from pipeline_data import FrameData
from pipeline_modules import Reader, Tracker, Saver
from video_source import VideoSource
from performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)


class IndependentInferencer:
    """独立推理器 — 逐帧从队列取出并推理"""

    def __init__(self, input_queue: Queue, output_queue: Queue,
                 inferencer, pipeline_id: str,
                 stop_event: threading.Event):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.inferencer = inferencer
        self.pipeline_id = pipeline_id
        self.stop_event = stop_event
        self.stats = {'total_frames': 0, 'inference_time': 0.0}

    def run(self):
        logger.info(f"[{self.pipeline_id}] IndependentInferencer started")

        try:
            while not self.stop_event.is_set():
                try:
                    frame_data = self.input_queue.get(timeout=5.0)
                except Empty:
                    continue

                if frame_data is None:
                    self.output_queue.put(None, timeout=1.0)
                    break

                start_time = time.time()
                try:
                    detections = self.inferencer.infer(frame_data.frame)
                    frame_data.detections = detections
                except Exception as e:
                    logger.error(f"[{self.pipeline_id}] Inference error: {e}")
                    frame_data.detections = []

                self.stats['inference_time'] += time.time() - start_time
                self.stats['total_frames'] += 1

                try:
                    self.output_queue.put(frame_data, timeout=5.0)
                except Exception as e:
                    logger.warning(f"[{self.pipeline_id}] Failed to put infer result: {e}")

        except Exception as e:
            logger.error(f"[{self.pipeline_id}] IndependentInferencer error: {e}")
        finally:
            logger.info(f"[{self.pipeline_id}] IndependentInferencer stopped. stats={self.stats}")


class IndependentPipeline:
    """单个视频的完整流水线"""

    def __init__(self, video_source: VideoSource, inferencer,
                 tracker_instance, save_func: Callable,
                 output_dir: str, pipeline_id: str,
                 stop_event: threading.Event, queue_size: int = 200):
        self.pipeline_id = pipeline_id
        self.stop_event = stop_event
        self.threads: List[threading.Thread] = []

        # 队列
        self.read_queue = Queue(maxsize=queue_size)
        self.infer_queue = Queue(maxsize=queue_size)
        self.track_queue = Queue(maxsize=queue_size)

        # Reader
        self.reader = Reader(
            video_source=video_source,
            output_queue=self.read_queue,
            pipeline_id=pipeline_id,
            stop_event=stop_event,
        )

        # Inferencer
        self.inferencer = IndependentInferencer(
            input_queue=self.read_queue,
            output_queue=self.infer_queue,
            inferencer=inferencer,
            pipeline_id=pipeline_id,
            stop_event=stop_event,
        )

        # Tracker
        self.tracker = Tracker(
            input_queue=self.infer_queue,
            output_queue=self.track_queue,
            tracker_instance=tracker_instance,
            pipeline_id=pipeline_id,
            stop_event=stop_event,
        )

        # Saver
        self.saver = Saver(
            input_queue=self.track_queue,
            output_dir=output_dir,
            save_func=save_func,
            pipeline_id=pipeline_id,
            stop_event=stop_event,
        )

    def start(self):
        for name, target in [
            (f"Reader_{self.pipeline_id}", self.reader.run),
            (f"Inferencer_{self.pipeline_id}", self.inferencer.run),
            (f"Tracker_{self.pipeline_id}", self.tracker.run),
            (f"Saver_{self.pipeline_id}", self.saver.run),
        ]:
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self.threads.append(t)

    def get_stats(self) -> Dict:
        return {
            'reader': self.reader.stats.copy(),
            'inferencer': self.inferencer.stats.copy(),
            'tracker': self.tracker.stats.copy(),
            'saver': self.saver.stats.copy(),
        }


class IndependentPipelineManager:
    """策略2顶层管理器 — 管理所有独立流水线"""

    def __init__(self,
                 video_sources: List,
                 inferencer_factory: Callable,
                 tracker_factory: Callable,
                 save_func: Callable,
                 output_dir: str = "result",
                 queue_size: int = 200):
        self.num_videos = len(video_sources)
        self.stop_event = threading.Event()
        self.pipelines: List[IndependentPipeline] = []

        for i, source in enumerate(video_sources):
            pipeline_id = f"video_{i}"
            inferencer = inferencer_factory(pipeline_id)
            tracker_instance = tracker_factory(pipeline_id)

            pipeline = IndependentPipeline(
                video_source=source,
                inferencer=inferencer,
                tracker_instance=tracker_instance,
                save_func=save_func,
                output_dir=output_dir,
                pipeline_id=pipeline_id,
                stop_event=self.stop_event,
                queue_size=queue_size,
            )
            self.pipelines.append(pipeline)

        logger.info(f"IndependentPipelineManager: {self.num_videos} pipelines created")

    def start(self):
        logger.info("Starting IndependentPipelineManager...")
        for pipeline in self.pipelines:
            pipeline.start()
        total_threads = sum(len(p.threads) for p in self.pipelines)
        logger.info(f"Started {total_threads} threads across {self.num_videos} pipelines")

    def stop(self):
        logger.info("Stopping IndependentPipelineManager...")
        self.stop_event.set()

    def wait(self, timeout: Optional[float] = None) -> bool:
        all_done = True
        for pipeline in self.pipelines:
            for t in pipeline.threads:
                t.join(timeout=timeout if timeout else 5.0)
                if t.is_alive():
                    all_done = False
                    logger.warning(f"Thread {t.name} still alive")
        return all_done

    def get_stats(self) -> Dict:
        return {
            'num_videos': self.num_videos,
            'pipelines': [p.get_stats() for p in self.pipelines],
        }
