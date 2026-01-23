"""
视频源抽象层
支持本地视频文件和网络直播源
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)


class VideoSource(ABC):
    """
    视频源抽象基类
    
    所有视频源（本地文件、Webcam、RTSP直播等）都应继承此类
    """
    
    def __init__(self, source_id: str, name: str = None):
        """
        Args:
            source_id: 视频源唯一标识（可以是文件路径或URL）
            name: 可读的名称，用于日志和输出目录
        """
        self.source_id = source_id
        self.name = name or source_id
        self._is_open = False
    
    @abstractmethod
    def open(self) -> bool:
        """
        打开视频源
        
        Returns:
            bool: 是否成功打开
        """
        pass
    
    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        读取一帧
        
        Returns:
            (success, frame): success为True表示成功读取，frame为BGR格式的np.ndarray
        """
        pass
    
    @abstractmethod
    def get_fps(self) -> float:
        """
        获取视频帧率
        
        Returns:
            float: 每秒帧数，若未知返回30.0
        """
        pass
    
    @abstractmethod
    def get_frame_count(self) -> int:
        """
        获取总帧数
        
        Returns:
            int: 总帧数，若未知（如直播流）返回-1
        """
        pass
    
    @abstractmethod
    def get_resolution(self) -> Tuple[int, int]:
        """
        获取视频分辨率
        
        Returns:
            (width, height): 视频宽高
        """
        pass
    
    @abstractmethod
    def close(self):
        """关闭视频源，释放资源"""
        pass
    
    @property
    def is_open(self) -> bool:
        """视频源是否已打开"""
        return self._is_open
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class LocalVideoSource(VideoSource):
    """
    本地视频文件源
    支持MP4、AVI、MKV等OpenCV支持的格式
    """
    
    def __init__(self, file_path: str):
        """
        Args:
            file_path: 本地视频文件路径
        """
        super().__init__(file_path, f"LocalVideo({file_path})")
        self.file_path = file_path
        self.cap = None
    
    def open(self) -> bool:
        """打开本地视频文件"""
        try:
            self.cap = cv2.VideoCapture(self.file_path)
            if not self.cap.isOpened():
                logger.error(f"Failed to open video file: {self.file_path}")
                return False
            
            self._is_open = True
            logger.info(f"Opened video file: {self.file_path}")
            logger.info(f"  Resolution: {self.get_resolution()}")
            logger.info(f"  FPS: {self.get_fps()}")
            logger.info(f"  Total frames: {self.get_frame_count()}")
            return True
        except Exception as e:
            logger.error(f"Error opening video file {self.file_path}: {e}")
            return False
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """读取一帧"""
        if not self._is_open or self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        if not ret:
            logger.debug(f"Reached end of video file: {self.file_path}")
        return ret, frame
    
    def get_fps(self) -> float:
        """获取视频帧率"""
        if not self._is_open or self.cap is None:
            return 30.0
        
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return fps if fps > 0 else 30.0
    
    def get_frame_count(self) -> int:
        """获取总帧数"""
        if not self._is_open or self.cap is None:
            return -1
        
        count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return count if count > 0 else -1
    
    def get_resolution(self) -> Tuple[int, int]:
        """获取视频分辨率"""
        if not self._is_open or self.cap is None:
            return (640, 480)
        
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height) if width > 0 and height > 0 else (640, 480)
    
    def close(self):
        """关闭视频文件"""
        if self.cap is not None:
            self.cap.release()
            self._is_open = False
            logger.info(f"Closed video file: {self.file_path}")


class WebcamSource(VideoSource):
    """
    网络摄像头/Webcam源
    支持：
    - 本地摄像头设备 (0, 1, 2...)
    - RTSP直播流 (rtsp://...)
    - HTTP直播流 (http://...)
    - MJPEG直播流
    """
    
    def __init__(self, source: str | int):
        """
        Args:
            source: 摄像头标识
                   - int类型：本地设备ID (0为默认摄像头)
                   - str类型：URL地址 (rtsp://, http://, etc.)
        """
        if isinstance(source, int):
            source_name = f"LocalCamera({source})"
        else:
            source_name = f"WebcamStream({source})"
        
        super().__init__(str(source), source_name)
        self.source = source
        self.cap = None
        self._resolution = (640, 480)
        self._fps = 30.0
    
    def open(self) -> bool:
        """打开网络摄像头或直播流"""
        try:
            # 转换source为适合VideoCapture的格式
            if isinstance(self.source, int):
                # 本地摄像头
                cap_source = self.source
            else:
                # URL格式，直接使用
                cap_source = self.source
            
            self.cap = cv2.VideoCapture(cap_source)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open webcam source: {self.source}")
                return False
            
            # 设置缓冲区大小（减少延迟）
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # 尝试读取一帧以验证连接
            ret, frame = self.cap.read()
            if not ret:
                logger.error(f"Cannot read from webcam source: {self.source}")
                self.cap.release()
                return False
            
            self._is_open = True
            self._resolution = (frame.shape[1], frame.shape[0])
            
            logger.info(f"Opened webcam source: {self.source}")
            logger.info(f"  Resolution: {self._resolution}")
            return True
        
        except Exception as e:
            logger.error(f"Error opening webcam source {self.source}: {e}")
            return False
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """读取一帧"""
        if not self._is_open or self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        if not ret:
            logger.warning(f"Failed to read from webcam source: {self.source}")
        return ret, frame
    
    def get_fps(self) -> float:
        """获取帧率"""
        if not self._is_open or self.cap is None:
            return 30.0
        
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return fps if fps > 0 else self._fps
    
    def get_frame_count(self) -> int:
        """
        获取总帧数
        对于实时流，返回-1（未知）
        """
        return -1  # 实时流不知道总帧数
    
    def get_resolution(self) -> Tuple[int, int]:
        """获取分辨率"""
        if not self._is_open or self.cap is None:
            return self._resolution
        
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if width > 0 and height > 0:
            self._resolution = (width, height)
        
        return self._resolution
    
    def close(self):
        """关闭网络摄像头"""
        if self.cap is not None:
            self.cap.release()
            self._is_open = False
            logger.info(f"Closed webcam source: {self.source}")
