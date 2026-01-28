"""
Pipeline中使用的数据结构
"""

from dataclasses import dataclass
from typing import Optional, List, Any
import numpy as np


@dataclass
class FrameData:
    """流经Pipeline的帧数据结构"""
    frame_id: int                      # 帧序号（1-based）
    timestamp: float                   # 时间戳
    frame: np.ndarray                  # 原始帧图像 (BGR格式)
    video_id: str                      # 视频源ID
    video_name: str                    # 视频源名称

    # 批处理相关
    video_index: int = 0               # 视频索引（用于批处理系统）

    # 推理结果
    detections: Optional[List[Any]] = None


@dataclass
class PipelineStatistics:
    """Pipeline的统计信息"""
    pipeline_id: str
    total_frames: int = 0
    dropped_frames: int = 0
    inference_time: float = 0.0
    tracking_time: float = 0.0
    average_inference_fps: float = 0.0
    average_tracking_fps: float = 0.0
    processing_state: str = "idle"
    error_message: Optional[str] = None
