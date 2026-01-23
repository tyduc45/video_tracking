"""
Pipeline各个处理阶段的实现
"""

import threading
import logging
import time
from queue import Queue
from typing import Callable, Optional
import numpy as np

from pipeline_data import FrameData, PipelineStatistics
from video_source import VideoSource

logger = logging.getLogger(__name__)


class Reader:
    """
    阶段1：视频读取器（生产者）
    从VideoSource读取帧，放入队列供下一阶段使用
    """
    
    def __init__(self, video_source: VideoSource, output_queue: Queue, 
                 pipeline_id: str, stop_event: threading.Event):
        """
        Args:
            video_source: 视频源
            output_queue: 输出队列（到Inferencer）
            pipeline_id: Pipeline标识
            stop_event: 停止信号
        """
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
                    self.stop_event.set()
                    break
                
                frame_id += 1
                
                frame_data = FrameData(
                    frame_id=frame_id,
                    timestamp=time.time(),
                    frame=frame,
                    video_id=self.pipeline_id,
                    video_name=self.video_source.name,
                )
                
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
            # 发送结束标志
            try:
                self.output_queue.put(None, timeout=1.0)
            except:
                pass
            logger.info(f"[{self.pipeline_id}] Reader stopped")


class Inferencer:
    """
    阶段2：推理器（消费者+生产者）
    读取帧进行YOLO推理，输出检测结果
    """
    
    def __init__(self, input_queue: Queue, output_queue: Queue,
                 inference_func: Callable, pipeline_id: str, 
                 stop_event: threading.Event):
        """
        Args:
            input_queue: 输入队列（来自Reader）
            output_queue: 输出队列（到Tracker）
            inference_func: 推理函数 (frame) -> detections
            pipeline_id: Pipeline标识
            stop_event: 停止信号
        """
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.inference_func = inference_func
        self.pipeline_id = pipeline_id
        self.stop_event = stop_event
        
        self.stats = {
            'total_frames': 0,
            'inference_time': 0.0,
        }
    
    def run(self):
        """执行推理循环"""
        logger.info(f"[{self.pipeline_id}] Inferencer started")
        
        try:
            while not self.stop_event.is_set():
                try:
                    frame_data = self.input_queue.get(timeout=5.0)
                    
                    if frame_data is None:
                        # 收到结束标志
                        logger.info(f"[{self.pipeline_id}] Inferencer received end signal")
                        self.output_queue.put(None, timeout=1.0)
                        break
                    
                    # 执行推理
                    start_time = time.time()
                    detections = self.inference_func(frame_data.frame)
                    inference_time = time.time() - start_time
                    
                    frame_data.detections = detections
                    
                    self.stats['total_frames'] += 1
                    self.stats['inference_time'] += inference_time
                    
                    # 输出到下一队列
                    self.output_queue.put(frame_data, timeout=5.0)
                
                except Exception as e:
                    if "Empty" not in str(type(e).__name__):
                        logger.warning(f"[{self.pipeline_id}] Inferencer queue timeout or error: {e}")
        
        except Exception as e:
            logger.error(f"[{self.pipeline_id}] Inferencer error: {e}")
            self.stop_event.set()
        
        finally:
            logger.info(f"[{self.pipeline_id}] Inferencer stopped")


class Tracker:
    """
    阶段3：追踪器（消费者+生产者）
    读取推理结果，进行ByteTrack追踪
    """
    
    def __init__(self, input_queue: Queue, output_queue: Queue,
                 tracker_instance, pipeline_id: str,
                 stop_event: threading.Event):
        """
        Args:
            input_queue: 输入队列（来自Inferencer）
            output_queue: 输出队列（到Saver）
            tracker_instance: ByteTrack追踪器实例
            pipeline_id: Pipeline标识
            stop_event: 停止信号
        """
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
                        # 收到结束标志
                        logger.info(f"[{self.pipeline_id}] Tracker received end signal")
                        self.output_queue.put(None, timeout=1.0)
                        break
                    
                    # 执行追踪
                    start_time = time.time()
                    
                    # 这里需要根据实际的ByteTrack API调用追踪
                    # 暂时为占位符
                    if frame_data.detections is not None:
                        tracks = self.tracker.update(frame_data.detections)
                        frame_data.tracks = tracks
                    
                    tracking_time = time.time() - start_time
                    
                    self.stats['total_frames'] += 1
                    self.stats['tracking_time'] += tracking_time
                    
                    # 输出到下一队列
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
    """
    阶段4：结果保存器（消费者）
    读取追踪结果，保存输出帧和生成最终视频
    """
    
    def __init__(self, input_queue: Queue, output_dir: str,
                 save_func: Callable, pipeline_id: str,
                 stop_event: threading.Event):
        """
        Args:
            input_queue: 输入队列（来自Tracker）
            output_dir: 输出目录
            save_func: 保存函数 (frame_data, output_dir) -> None
            pipeline_id: Pipeline标识
            stop_event: 停止信号
        """
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
                        # 收到结束标志
                        logger.info(f"[{self.pipeline_id}] Saver received end signal")
                        break
                    
                    # 执行保存
                    self.save_func(frame_data, self.output_dir)
                    self.stats['total_frames'] += 1
                
                except Exception as e:
                    if "Empty" not in str(type(e).__name__):
                        logger.warning(f"[{self.pipeline_id}] Saver queue timeout: {e}")
        
        except Exception as e:
            logger.error(f"[{self.pipeline_id}] Saver error: {e}")
            self.stop_event.set()
        
        finally:
            logger.info(f"[{self.pipeline_id}] Saver stopped")
