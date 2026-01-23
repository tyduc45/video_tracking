"""
单一视频流水线 - Pipeline的核心组件
"""

import threading
import logging
import time
from queue import Queue
from typing import Callable, Optional
from datetime import datetime

from video_source import VideoSource
from pipeline_data import FrameData, PipelineStatistics
from pipeline_modules import Reader, Inferencer, Tracker, Saver

logger = logging.getLogger(__name__)


class SingleVideoPipeline:
    """
    单一视频的完整处理流水线
    
    架构:
    ┌─────────────┐
    │   Reader    │  (Producer)   从视频源读取帧
    │   (Thread)  │
    └──────┬──────┘
           │ FrameData
        Queue1
           │
    ┌──────▼──────────┐
    │ Inferencer      │  (Consumer+Producer) YOLO推理
    │ (Thread)        │
    └──────┬──────────┘
           │ FrameData + Detections
        Queue2
           │
    ┌──────▼──────────┐
    │ Tracker         │  (Consumer+Producer) ByteTrack追踪
    │ (Thread)        │
    └──────┬──────────┘
           │ FrameData + Tracks
        Queue3
           │
    ┌──────▼──────────┐
    │ Saver           │  (Consumer) 保存结果
    │ (Thread)        │
    └─────────────────┘
    """
    
    def __init__(self, pipeline_id: str, video_source: VideoSource,
                 inference_func: Callable = None,
                 tracker_instance = None,
                 save_func: Callable = None,
                 output_dir: str = "result",
                 queue_size: int = 10):
        """
        初始化Pipeline
        
        Args:
            pipeline_id: Pipeline唯一标识（如"video_0", "camera_1"）
            video_source: VideoSource实例
            inference_func: 推理函数 (frame: np.ndarray) -> detections
            tracker_instance: ByteTrack追踪器实例
            save_func: 保存函数 (frame_data: FrameData, output_dir: str) -> None
            output_dir: 输出目录
            queue_size: 队列大小限制
        """
        self.pipeline_id = pipeline_id
        self.video_source = video_source
        self.inference_func = inference_func
        self.tracker_instance = tracker_instance
        self.save_func = save_func
        self.output_dir = output_dir
        self.queue_size = queue_size
        
        # 三个连接各阶段的队列
        self.queue1 = Queue(maxsize=queue_size)  # Reader -> Inferencer
        self.queue2 = Queue(maxsize=queue_size)  # Inferencer -> Tracker
        self.queue3 = Queue(maxsize=queue_size)  # Tracker -> Saver
        
        # 停止信号
        self.stop_event = threading.Event()
        
        # 四个处理线程
        self.reader = None
        self.inferencer = None
        self.tracker = None
        self.saver = None
        
        self.reader_thread = None
        self.inferencer_thread = None
        self.tracker_thread = None
        self.saver_thread = None
        
        # 统计信息
        self.stats = PipelineStatistics(pipeline_id=pipeline_id)
        self.start_time = None
        self.end_time = None
        
        logger.debug(f"[{self.pipeline_id}] Pipeline created")
    
    def _create_modules(self):
        """创建所有处理模块"""
        # Reader
        self.reader = Reader(
            video_source=self.video_source,
            output_queue=self.queue1,
            pipeline_id=self.pipeline_id,
            stop_event=self.stop_event
        )
        
        # Inferencer
        self.inferencer = Inferencer(
            input_queue=self.queue1,
            output_queue=self.queue2,
            inference_func=self.inference_func or self._dummy_inference,
            pipeline_id=self.pipeline_id,
            stop_event=self.stop_event
        )
        
        # Tracker
        self.tracker = Tracker(
            input_queue=self.queue2,
            output_queue=self.queue3,
            tracker_instance=self.tracker_instance or self._dummy_tracker,
            pipeline_id=self.pipeline_id,
            stop_event=self.stop_event
        )
        
        # Saver
        self.saver = Saver(
            input_queue=self.queue3,
            output_dir=self.output_dir,
            save_func=self.save_func or self._dummy_save,
            pipeline_id=self.pipeline_id,
            stop_event=self.stop_event
        )
    
    @staticmethod
    def _dummy_inference(frame):
        """占位推理函数"""
        return None
    
    @staticmethod
    def _dummy_tracker():
        """占位追踪器"""
        class DummyTracker:
            def update(self, detections):
                return None
        return DummyTracker()
    
    @staticmethod
    def _dummy_save(frame_data, output_dir):
        """占位保存函数"""
        pass
    
    def start(self):
        """
        启动Pipeline
        创建所有工作线程并启动
        """
        if not self.reader:
            self._create_modules()
        
        logger.info(f"[{self.pipeline_id}] Starting pipeline...")
        self.stats.processing_state = "running"
        self.start_time = time.time()
        
        try:
            # 创建并启动四个工作线程
            self.reader_thread = threading.Thread(
                target=self.reader.run,
                name=f"{self.pipeline_id}-Reader",
                daemon=False
            )
            
            self.inferencer_thread = threading.Thread(
                target=self.inferencer.run,
                name=f"{self.pipeline_id}-Inferencer",
                daemon=False
            )
            
            self.tracker_thread = threading.Thread(
                target=self.tracker.run,
                name=f"{self.pipeline_id}-Tracker",
                daemon=False
            )
            
            self.saver_thread = threading.Thread(
                target=self.saver.run,
                name=f"{self.pipeline_id}-Saver",
                daemon=False
            )
            
            # 按顺序启动线程（从上游到下游）
            self.reader_thread.start()
            self.inferencer_thread.start()
            self.tracker_thread.start()
            self.saver_thread.start()
            
            logger.info(f"[{self.pipeline_id}] Pipeline started successfully")
        
        except Exception as e:
            logger.error(f"[{self.pipeline_id}] Failed to start pipeline: {e}")
            self.stats.processing_state = "failed"
            self.stats.error_message = str(e)
            self.stop()
    
    def stop(self):
        """
        停止Pipeline
        发送停止信号，等待所有线程完成
        """
        logger.info(f"[{self.pipeline_id}] Stopping pipeline...")
        self.stop_event.set()
        
        # 清空队列（避免线程阻塞）
        while not self.queue1.empty():
            try:
                self.queue1.get_nowait()
            except:
                break
        
        while not self.queue2.empty():
            try:
                self.queue2.get_nowait()
            except:
                break
        
        while not self.queue3.empty():
            try:
                self.queue3.get_nowait()
            except:
                break
    
    def wait(self, timeout: Optional[float] = None):
        """
        等待Pipeline完成
        
        Args:
            timeout: 最大等待时间（秒），None表示无限等待
        
        Returns:
            bool: 是否成功完成（True=完成，False=超时）
        """
        logger.debug(f"[{self.pipeline_id}] Waiting for pipeline completion...")
        
        if self.reader_thread:
            self.reader_thread.join(timeout=timeout)
        if self.inferencer_thread:
            self.inferencer_thread.join(timeout=timeout)
        if self.tracker_thread:
            self.tracker_thread.join(timeout=timeout)
        if self.saver_thread:
            self.saver_thread.join(timeout=timeout)
        
        self.end_time = time.time()
        
        # 检查所有线程是否完成
        all_done = (
            (not self.reader_thread or not self.reader_thread.is_alive()) and
            (not self.inferencer_thread or not self.inferencer_thread.is_alive()) and
            (not self.tracker_thread or not self.tracker_thread.is_alive()) and
            (not self.saver_thread or not self.saver_thread.is_alive())
        )
        
        if all_done:
            self.stats.processing_state = "completed"
            logger.info(f"[{self.pipeline_id}] Pipeline completed")
        else:
            self.stats.processing_state = "timeout"
            logger.warning(f"[{self.pipeline_id}] Pipeline wait timeout")
        
        return all_done
    
    def get_statistics(self) -> PipelineStatistics:
        """获取Pipeline统计信息"""
        if self.reader:
            self.stats.total_frames = self.reader.stats.get('total_frames', 0)
            self.stats.dropped_frames = self.reader.stats.get('dropped_frames', 0)
        
        if self.inferencer:
            inference_time = self.inferencer.stats.get('inference_time', 0.0)
            inference_frames = self.inferencer.stats.get('total_frames', 0)
            self.stats.inference_time = inference_time
            if inference_frames > 0:
                self.stats.average_inference_fps = inference_frames / max(inference_time, 0.001)
        
        if self.tracker:
            tracking_time = self.tracker.stats.get('tracking_time', 0.0)
            tracking_frames = self.tracker.stats.get('total_frames', 0)
            self.stats.tracking_time = tracking_time
            if tracking_frames > 0:
                self.stats.average_tracking_fps = tracking_frames / max(tracking_time, 0.001)
        
        return self.stats
    
    def print_statistics(self):
        """打印Pipeline统计信息"""
        stats = self.get_statistics()
        
        elapsed_time = (self.end_time - self.start_time) if (self.start_time and self.end_time) else 0
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Pipeline Statistics: {stats.pipeline_id}")
        logger.info(f"{'='*60}")
        logger.info(f"Total frames:          {stats.total_frames}")
        logger.info(f"Dropped frames:        {stats.dropped_frames}")
        logger.info(f"Processing state:      {stats.processing_state}")
        logger.info(f"Total elapsed time:    {elapsed_time:.2f}s")
        logger.info(f"Inference time:        {stats.inference_time:.2f}s (FPS: {stats.average_inference_fps:.1f})")
        logger.info(f"Tracking time:         {stats.tracking_time:.2f}s (FPS: {stats.average_tracking_fps:.1f})")
        if stats.error_message:
            logger.info(f"Error:                 {stats.error_message}")
        logger.info(f"{'='*60}\n")
