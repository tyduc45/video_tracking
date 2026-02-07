"""
Pipeline各个处理阶段的实现
"""

import threading
import logging
import time
from queue import Queue
from typing import Callable

from pipeline_data import FrameData
from video_source import VideoSource
from performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)


class Reader:
    """视频读取器（生产者）- 从VideoSource读取帧"""

    def __init__(self, video_source: VideoSource, output_queue: Queue,
                 pipeline_id: str, stop_event: threading.Event):
        self.video_source = video_source
        self.output_queue = output_queue
        self.pipeline_id = pipeline_id
        self.stop_event = stop_event

        self.stats = {
            'total_frames': 0,
            'dropped_frames': 0,
        }

    def run(self):
        """执行读取循环"""
        logger.info(f"[{self.pipeline_id}] Reader started")

        if not self.video_source.open():
            logger.error(f"[{self.pipeline_id}] Failed to open video source")
            self.stop_event.set()
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

                # 性能探针: 帧开始处理
                PerformanceMonitor.probe(self.pipeline_id, frame_id, "start")

                try:
                    self.output_queue.put(frame_data, timeout=5.0)
                    self.stats['total_frames'] += 1
                except Exception as e:
                    logger.warning(f"[{self.pipeline_id}] Failed to put frame in queue: {e}")
                    self.stats['dropped_frames'] += 1

        except Exception as e:
            logger.error(f"[{self.pipeline_id}] Reader error: {e}")
            self.stop_event.set()

        finally:
            self.video_source.close()
            try:
                self.output_queue.put(None, timeout=1.0)
            except:
                pass
            logger.info(f"[{self.pipeline_id}] Reader stopped")


class Tracker:
    """追踪器（消费者+生产者）- 进行追踪处理"""

    def __init__(self, input_queue: Queue, output_queue: Queue,
                 tracker_instance, pipeline_id: str,
                 stop_event: threading.Event):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.tracker = tracker_instance
        self.pipeline_id = pipeline_id
        self.stop_event = stop_event

        self.stats = {
            'total_frames': 0,
            'tracking_time': 0.0,
        }

    def run(self):
        """执行追踪循环"""
        logger.info(f"[{self.pipeline_id}] Tracker started")

        try:
            while not self.stop_event.is_set():
                try:
                    frame_data = self.input_queue.get(timeout=5.0)

                    if frame_data is None:
                        logger.info(f"[{self.pipeline_id}] Tracker received end signal")
                        self.output_queue.put(None, timeout=1.0)
                        break

                    start_time = time.time()

                    if frame_data.frame is not None and self.tracker is not None:
                        track_results = self.tracker.update(frame_data.frame, frame_data.detections)
                        if track_results is not None:
                            frame_data.detections = track_results

                    tracking_time = time.time() - start_time

                    self.stats['total_frames'] += 1
                    self.stats['tracking_time'] += tracking_time

                    self.output_queue.put(frame_data, timeout=5.0)

                except Exception as e:
                    if "Empty" not in str(type(e).__name__):
                        logger.warning(f"[{self.pipeline_id}] Tracker queue timeout: {e}")

        except Exception as e:
            logger.error(f"[{self.pipeline_id}] Tracker error: {e}")
            self.stop_event.set()

        finally:
            logger.info(f"[{self.pipeline_id}] Tracker stopped")


class Saver:
    """结果保存器（消费者）- 保存输出帧"""

    def __init__(self, input_queue: Queue, output_dir: str,
                 save_func: Callable, pipeline_id: str,
                 stop_event: threading.Event):
        self.input_queue = input_queue
        self.output_dir = output_dir
        self.save_func = save_func
        self.pipeline_id = pipeline_id
        self.stop_event = stop_event

        self.stats = {
            'total_frames': 0,
        }

    def run(self):
        """执行保存循环"""
        logger.info(f"[{self.pipeline_id}] Saver started")

        try:
            while not self.stop_event.is_set():
                try:
                    frame_data = self.input_queue.get(timeout=5.0)

                    if frame_data is None:
                        logger.info(f"[{self.pipeline_id}] Saver received end signal")
                        break

                    self.save_func(frame_data, self.output_dir)
                    self.stats['total_frames'] += 1

                except Exception as e:
                    if "Empty" not in str(type(e).__name__):
                        logger.warning(f"[{self.pipeline_id}] Saver queue timeout: {e}")

        except Exception as e:
            logger.error(f"[{self.pipeline_id}] Saver error: {e}")
            self.stop_event.set()

        finally:
            # 性能探针: 视频处理结束
            PerformanceMonitor.probe(self.pipeline_id, -1, "finish")
            logger.info(f"[{self.pipeline_id}] Saver stopped")
