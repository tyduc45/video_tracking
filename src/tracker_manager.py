"""
ByteTrack 追踪器管理系统，实现了多视频源的序列号同步机制
"""
import heapq
import threading
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass

# 尝试导入 ByteTrack
try:
    from ultralytics import YOLO
    # ByteTrack 支持通过 ultralytics 使用
    HAS_BYTETRACK = True
except ImportError:
    HAS_BYTETRACK = False


@dataclass
class FrameData:
    """
    帧数据结构
    """
    frame: np.ndarray
    path: str
    video_id: str  # 视频源标识
    frame_id: int  # 帧序列号（1-based）
    detections: Optional[list] = None  # YOLO推理结果
    
    def __lt__(self, other):
        """用于优先队列比较，按帧序列号排序"""
        return self.frame_id < other.frame_id


class SingleVideoTracker:
    """
    单个视频源的追踪器，包含：
    - ByteTrack 跟踪器实例
    - 帧计数器（期望的下一帧编号）
    - 优先队列（用于乱序帧）
    - 缓冲区（已排序的帧）
    """
    
    def __init__(self, video_id: str, model_name: str = "yolov8n-pose.pt"):
        self.video_id = video_id
        self.counter = 1  # 期望的下一帧编号，初始化为 1
        self.priority_queue = []  # 小顶堆，存放乱序的帧
        self.buffer = []  # 输出缓冲区，存放已排序的帧
        self.lock = threading.Lock()  # 线程安全锁
        
        # 初始化 ByteTrack 跟踪器
        # 这里使用 YOLO 模型的 track 方法
        self.tracker_initialized = False
        self.model = None
        
    def _init_tracker(self, model_path: str):
        """初始化跟踪器（延迟初始化）"""
        if not self.tracker_initialized:
            try:
                self.model = YOLO(model_path)
                self.tracker_initialized = True
                print(f"[Tracker {self.video_id}] ByteTrack 初始化完成")
            except Exception as e:
                print(f"[Tracker {self.video_id}] 初始化失败: {e}")
    
    def process_frame(self, frame_data: FrameData, model_path: str = None) -> Optional[FrameData]:
        """
        处理单个帧，实现乱序恢复逻辑
        
        返回值：如果返回列表，表示有多个帧已排序完成
        """
        with self.lock:
            if model_path and not self.tracker_initialized:
                self._init_tracker(model_path)
            
            # 第一步：根据帧编号判断
            if frame_data.frame_id != self.counter:
                # 帧乱序，进入优先队列
                heapq.heappush(self.priority_queue, frame_data)
                return []
            
            # 第二步：帧编号匹配，直接进入缓冲区
            ready_frames = [frame_data]
            self.counter += 1
            
            # 第三步：检查优先队列
            while self.priority_queue and self.priority_queue[0].frame_id == self.counter:
                next_frame = heapq.heappop(self.priority_queue)
                ready_frames.append(next_frame)
                self.counter += 1
            
            return ready_frames
    
    def track_frames(self, frames: List[FrameData], model_path: str = None) -> List[FrameData]:
        """
        对已排序的帧进行追踪，并返回带有追踪信息的帧
        """
        if model_path and not self.tracker_initialized:
            self._init_tracker(model_path)
        
        tracked_frames = []
        
        for frame_data in frames:
            try:
                # 使用 YOLO 的 track 方法进行追踪
                if self.model:
                    results = self.model.track(
                        frame_data.frame,
                        persist=True,
                        verbose=False
                    )
                    
                    # 提取追踪结果
                    if results and len(results) > 0:
                        frame_data.detections = results[0]
                        # 可以从 results[0].boxes 获取检测框
                        # 从 results[0].boxes.id 获取追踪 ID
                
                tracked_frames.append(frame_data)
                
            except Exception as e:
                print(f"[Tracker {self.video_id}] 追踪帧 {frame_data.frame_id} 失败: {e}")
                tracked_frames.append(frame_data)
        
        return tracked_frames
    
    def get_buffer_size(self) -> int:
        """获取缓冲区大小"""
        with self.lock:
            return len(self.buffer)
    
    def get_queue_size(self) -> int:
        """获取优先队列大小"""
        with self.lock:
            return len(self.priority_queue)
    
    def get_expected_frame_id(self) -> int:
        """获取期望的下一帧编号"""
        with self.lock:
            return self.counter


class TrackerManager:
    """
    管理多个追踪器的全局管理器
    """
    
    def __init__(self, video_paths: List[str]):
        """
        初始化多个追踪器，每个视频源对应一个追踪器
        
        Args:
            video_paths: 视频源路径列表
        """
        self.trackers: Dict[str, SingleVideoTracker] = {}
        self.video_paths = video_paths
        self.lock = threading.Lock()
        
        # 创建追踪器实例
        for idx, path in enumerate(video_paths):
            video_id = f"video_{idx}"
            self.trackers[video_id] = SingleVideoTracker(video_id)
        
        print(f"[TrackerManager] 初始化了 {len(self.trackers)} 个追踪器")
    
    def get_video_id(self, path: str) -> str:
        """根据视频路径获取视频 ID"""
        try:
            idx = self.video_paths.index(path)
            return f"video_{idx}"
        except ValueError:
            return None
    
    def process_frame(self, frame_data: FrameData, model_path: str = None) -> List[FrameData]:
        """
        处理一个帧，分配给对应的追踪器
        
        Returns:
            已排序完成的帧列表
        """
        with self.lock:
            video_id = frame_data.video_id
            if video_id not in self.trackers:
                print(f"[TrackerManager] 警告：未知的视频 ID {video_id}")
                return []
            
            tracker = self.trackers[video_id]
            return tracker.process_frame(frame_data, model_path)
    
    def track_frames(self, frames: List[FrameData], model_path: str = None) -> List[FrameData]:
        """
        对已排序的帧进行追踪
        """
        result_frames = []
        
        for frame_data in frames:
            video_id = frame_data.video_id
            if video_id not in self.trackers:
                result_frames.append(frame_data)
                continue
            
            tracker = self.trackers[video_id]
            tracked = tracker.track_frames([frame_data], model_path)
            result_frames.extend(tracked)
        
        return result_frames
    
    def get_status(self) -> Dict:
        """获取所有追踪器的状态"""
        status = {}
        for video_id, tracker in self.trackers.items():
            status[video_id] = {
                "expected_frame_id": tracker.get_expected_frame_id(),
                "queue_size": tracker.get_queue_size(),
                "buffer_size": tracker.get_buffer_size()
            }
        return status
    
    def print_status(self):
        """打印所有追踪器的状态"""
        status = self.get_status()
        print("\n=== 追踪器状态 ===")
        for video_id, info in status.items():
            print(f"{video_id}: 期望帧={info['expected_frame_id']}, "
                  f"队列={info['queue_size']}, 缓冲={info['buffer_size']}")
        print("=================\n")
    
    def flush_all_buffers(self, model_path: str = None) -> List[FrameData]:
        """
        刷新所有追踪器的缓冲区，获取所有未输出的帧
        用于程序结束时确保所有帧都被处理
        
        Returns:
            所有缓冲的帧列表
        """
        all_frames = []
        
        for video_id, tracker in self.trackers.items():
            with tracker.lock:
                # 清空优先队列中所有的帧（即使是乱序的，也要输出）
                while tracker.priority_queue:
                    frame = heapq.heappop(tracker.priority_queue)
                    all_frames.append(frame)
                
                # 按帧编号排序所有收集的帧
                all_frames.sort(key=lambda x: x.frame_id)
        
        # 对所有收集的帧进行追踪
        if all_frames:
            all_frames = self.track_frames(all_frames, model_path)
        
        return all_frames
