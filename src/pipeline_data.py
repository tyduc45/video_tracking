"""
Pipeline中使用的数据结构
"""

from dataclasses import dataclass
from typing import Optional, List, Any
import numpy as np


@dataclass
class FrameData:
    """
    流经Pipeline的帧数据结构
    """
    frame_id: int                      # 帧序号（1-based）
    timestamp: float                   # 时间戳
    frame: np.ndarray                  # 原始帧图像 (BGR格式)
    video_id: str                      # 视频源ID
    video_name: str                    # 视频源名称
    
    # 推理结果
    detections: Optional[List[Any]] = None  # YOLO检测结果
    
    # 追踪结果
    tracks: Optional[List[Any]] = None      # ByteTrack追踪结果
    tracked_frame: Optional[np.ndarray] = None  # 绘制后的帧


@dataclass
class PipelineStatistics:
    """
    Pipeline的统计信息
    """
    pipeline_id: str                   # Pipeline ID
    total_frames: int = 0              # 处理的总帧数
    dropped_frames: int = 0            # 丢失的帧数
    inference_time: float = 0.0        # 推理总耗时(秒)
    tracking_time: float = 0.0         # 追踪总耗时(秒)
    average_inference_fps: float = 0.0 # 平均推理FPS
    average_tracking_fps: float = 0.0  # 平均追踪FPS
    processing_state: str = "idle"     # 处理状态: idle, running, completed, failed
    error_message: Optional[str] = None # 错误信息
