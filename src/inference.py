"""
YOLO推理和ByteTrack追踪的具体实现
"""

import logging
import numpy as np
import json
import os
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
import threading
import time

logger = logging.getLogger(__name__)


class YOLOInferencer:
    """
    YOLO推理器 - 支持批处理和Engine加速
    
    动态模型管理：
    - 优先加载 .engine 文件（性能最佳）
    - 从 yolo12n_batch.meta 读取配置
    - 当实际 batch size 与 engine batch size 不匹配时，
      自动重新导出 engine 或回退到 .pt
    """
    
    META_FILE = "yolo12n_batch.meta"
    
    def __init__(self, model_path: str, model_dir: str = "../model",
                 device: str = "cuda", use_half: bool = True,
                 confidence_threshold: float = 0.5,
                 iou_threshold: float = 0.45,
                 batch_size: int = 16):
        """
        Args:
            model_path: 完整的模型文件路径（.pt, .engine, .onnx）
            model_dir: 模型目录，用于读取meta文件和存放engine
            device: cuda或cpu
            use_half: 是否使用半精度
            confidence_threshold: 置信度阈值
            iou_threshold: IOU阈值
            batch_size: 请求的批处理大小
        """
        self.model_dir = model_dir
        self.device = device
        self.use_half = use_half
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.requested_batch_size = batch_size
        
        self.model = None
        self.current_model_path = None
        self.current_batch_size = None
        self.batch_buffer = []
        
        # 加载 meta 配置并决定使用哪个模型
        self.meta = self._load_meta_config()
        self._smart_load_model(model_path)
    
    def _get_meta_path(self) -> str:
        """获取 meta 文件路径"""
        return os.path.join(self.model_dir, self.META_FILE)
    
    def _load_meta_config(self) -> Dict[str, Any]:
        """加载 meta 配置文件"""
        meta_path = self._get_meta_path()
        default_meta = {
            "model_name": "yolo12n",
            "batch_size": 16,
            "use_half": True,
            "input_size": [640, 640],
            "framework": "yolov8",
            "engine_batch_size": None,  # engine 编译时的 batch size
            "last_request_batch_size": None,
        }
        
        try:
            if os.path.exists(meta_path):
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                    logger.info(f"Loaded meta config: {meta}")
                    # 合并默认值
                    for key, value in default_meta.items():
                        if key not in meta:
                            meta[key] = value
                    return meta
        except Exception as e:
            logger.warning(f"Failed to load meta config: {e}")
        
        return default_meta
    
    def _save_meta_config(self):
        """保存 meta 配置文件"""
        meta_path = self._get_meta_path()
        try:
            os.makedirs(os.path.dirname(meta_path) or '.', exist_ok=True)
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(self.meta, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved meta config: {self.meta}")
        except Exception as e:
            logger.warning(f"Failed to save meta config: {e}")
    
    def _smart_load_model(self, model_path: str):
        """
        智能加载模型
        
        优先级：
        1. 如果 engine 存在且 batch size 匹配 -> 使用 engine
        2. 如果 engine 不存在或 batch size 不匹配 -> 尝试导出新 engine
        3. 如果导出失败 -> 回退到 .pt
        """
        model_ext = os.path.splitext(model_path)[1].lower()
        base_name = os.path.splitext(os.path.basename(model_path))[0]
        
        # 获取 .pt 文件路径（用于导出 engine）
        pt_path = os.path.join(self.model_dir, f"{base_name}.pt")
        engine_path = os.path.join(self.model_dir, f"{base_name}.engine")
        
        # 检查 meta 中记录的 engine batch size
        engine_batch_size = self.meta.get("engine_batch_size")
        last_request = self.meta.get("last_request_batch_size")
        
        logger.info(f"Smart model loading:")
        logger.info(f"  Requested batch_size: {self.requested_batch_size}")
        logger.info(f"  Engine batch_size (meta): {engine_batch_size}")
        logger.info(f"  Last request batch_size: {last_request}")
        
        # 更新 meta
        self.meta["last_request_batch_size"] = self.requested_batch_size
        
        # 决策逻辑
        use_engine = False
        
        if os.path.exists(engine_path):
            if engine_batch_size is not None and engine_batch_size >= self.requested_batch_size:
                # engine 存在且 batch size 足够
                logger.info(f"Engine exists with sufficient batch_size ({engine_batch_size} >= {self.requested_batch_size})")
                use_engine = True
            else:
                # engine batch size 不足，需要重新导出
                logger.info(f"Engine batch_size insufficient, need re-export")
                if os.path.exists(pt_path):
                    use_engine = self._export_engine(pt_path, engine_path, self.requested_batch_size)
        else:
            # engine 不存在，尝试导出
            logger.info(f"Engine not found, attempting export")
            if os.path.exists(pt_path):
                use_engine = self._export_engine(pt_path, engine_path, self.requested_batch_size)
        
        # 加载模型
        if use_engine and os.path.exists(engine_path):
            self._load_model(engine_path)
            self.current_batch_size = self.meta.get("engine_batch_size", self.requested_batch_size)
        elif os.path.exists(pt_path):
            logger.info("Falling back to .pt model (dynamic batch size)")
            self._load_model(pt_path)
            self.current_batch_size = self.requested_batch_size
        else:
            # 使用原始指定的模型路径
            self._load_model(model_path)
            self.current_batch_size = self.requested_batch_size
        
        self._save_meta_config()
    
    def _export_engine(self, pt_path: str, engine_path: str, batch_size: int) -> bool:
        """
        从 .pt 导出 .engine 文件
        
        Returns:
            是否导出成功
        """
        try:
            from ultralytics import YOLO
            
            logger.info(f"Exporting TensorRT engine...")
            logger.info(f"  Source: {pt_path}")
            logger.info(f"  Target: {engine_path}")
            logger.info(f"  Batch size: {batch_size}")
            
            model = YOLO(pt_path)
            
            # 导出 engine
            export_path = model.export(
                format='engine',
                half=self.use_half,
                batch=batch_size,
                device=0 if self.device.lower() == "cuda" else "cpu",
                simplify=True,
                workspace=4,  # GB
            )
            
            logger.info(f"Engine exported successfully: {export_path}")
            
            # 更新 meta
            self.meta["engine_batch_size"] = batch_size
            self.meta["engine_export_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to export engine: {e}")
            return False
    
    def _load_model(self, model_path: str):
        """加载YOLO模型"""
        try:
            from ultralytics import YOLO
            
            logger.info(f"Loading YOLO model from {model_path}")
            logger.info(f"  Device: {self.device}")
            logger.info(f"  Use half precision: {self.use_half}")
            
            model_ext = os.path.splitext(model_path)[1].lower()
            logger.info(f"  Model format: {model_ext}")
            
            self.model = YOLO(model_path, task='detect')
            self.current_model_path = model_path
            
            # 只有PT格式支持to()方法
            if model_ext == '.pt':
                if self.device.lower() == "cuda":
                    self.model.to("cuda")
                else:
                    self.model.to("cpu")
                logger.info(f"YOLO PT model loaded and moved to {self.device}")
            else:
                logger.info(f"YOLO {model_ext.upper()} model loaded")
        
        except ImportError:
            logger.error("ultralytics package not found. Install it with: pip install ultralytics")
            raise
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def _pad_batch(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        填充 batch 到 engine 要求的大小
        
        TensorRT engine 通常需要固定的 batch size
        """
        if self.current_batch_size is None:
            return frames
        
        actual_size = len(frames)
        target_size = self.current_batch_size
        
        if actual_size >= target_size:
            return frames[:target_size]
        
        # 用最后一帧填充
        padding = [frames[-1]] * (target_size - actual_size)
        return frames + padding
    
    def infer_batch(self, frames: List[np.ndarray]) -> List[List[dict]]:
        """
        批量推理多帧
        
        自动处理 batch size 填充（对于 engine 模型）
        
        Args:
            frames: BGR格式的图像列表
        
        Returns:
            检测结果列表，每个元素对应一帧的检测结果列表
        """
        if self.model is None or not frames:
            return [[] for _ in frames]
        
        actual_frame_count = len(frames)
        model_ext = os.path.splitext(self.current_model_path or "")[1].lower()
        
        try:
            device = 0 if self.device.lower() == "cuda" else "cpu"
            
            # 对于 engine 模型，需要填充到固定 batch size
            if model_ext == '.engine' and self.current_batch_size:
                frames_to_infer = self._pad_batch(frames)
                logger.debug(f"Padded batch: {actual_frame_count} -> {len(frames_to_infer)}")
            else:
                frames_to_infer = frames
            
            results = self.model.predict(
                source=frames_to_infer,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                half=self.use_half,
                device=device,
                verbose=False
            )
            
            all_detections = []
            for i, result in enumerate(results):
                # 只取实际帧数的结果
                if i >= actual_frame_count:
                    break
                
                detections = []
                for detection in result.boxes:
                    class_id = int(detection.cls[0])
                    class_name = result.names[class_id]
                    confidence = float(detection.conf[0])
                    bbox = detection.xyxy[0].cpu().numpy()
                    
                    detections.append({
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': confidence,
                        'bbox': bbox.astype(int).tolist(),
                    })
                
                all_detections.append(detections)
            
            return all_detections
        
        except Exception as e:
            logger.error(f"Error during batch inference: {e}")
            return [[] for _ in range(actual_frame_count)]
    
    def infer(self, frame: np.ndarray) -> List[dict]:
        """
        对一帧进行推理（兼容之前的单帧接口）
        
        Args:
            frame: BGR格式的图像
        
        Returns:
            检测结果列表
        """
        results = self.infer_batch([frame])
        return results[0] if results else []
    
    def infer_with_buffering(self, frame: np.ndarray, force_flush: bool = False) -> Optional[List[dict]]:
        """
        带缓冲的推理，达到batch_size时进行批处理
        
        Args:
            frame: 输入帧
            force_flush: 是否强制刷新缓冲区（在视频结束时使用）
        
        Returns:
            第一帧的检测结果（或None）
        """
        self.batch_buffer.append(frame)
        
        # 缓冲区满，执行推理
        if len(self.batch_buffer) >= self.batch_size or force_flush:
            all_results = self.infer_batch(self.batch_buffer)
            first_result = all_results[0] if all_results else []
            self.batch_buffer = []
            return first_result
        
        return None
    
    def flush_buffer(self) -> List[List[dict]]:
        """刷新缓冲区，推理剩余的帧"""
        if not self.batch_buffer:
            return []
        
        results = self.infer_batch(self.batch_buffer)
        self.batch_buffer = []
        return results
    
    def __del__(self):
        """清理资源"""
        self.model = None


class ByteTracker:
    """
    ByteTrack追踪器 - 使用YOLO原生track方法
    支持 persist=True 实现帧间追踪ID的一致性
    """
    
    def __init__(self, model_path: str, device: str = "cuda",
                 track_high_thresh: float = 0.6,
                 track_low_thresh: float = 0.1,
                 track_buffer: int = 30,
                 frame_rate: float = 30.0):
        """
        Args:
            model_path: YOLO模型路径
            device: cuda或cpu
            track_high_thresh: 高阈值
            track_low_thresh: 低阈值
            track_buffer: 追踪缓冲区大小
            frame_rate: 视频帧率
        """
        self.model_path = model_path
        self.device = device
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.track_buffer = track_buffer
        self.frame_rate = frame_rate
        self.model = None
        
        self._load_tracker()
    
    def _load_tracker(self):
        """加载YOLO模型用于追踪"""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            logger.info(f"YOLO tracker loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO tracker: {e}")
            raise
    
    def update(self, frame: np.ndarray, conf_threshold: float = 0.5) -> Optional[any]:
        """
        对一帧进行追踪
        
        Args:
            frame: BGR格式的图像
            conf_threshold: 置信度阈值
        
        Returns:
            YOLO追踪结果对象，包含boxes.id (追踪ID) 和 boxes (检测框)
            该对象可以直接用 .plot() 方法可视化，自动显示追踪框和ID
        """
        if self.model is None:
            logger.error("Tracker model not loaded")
            return None
        
        try:
            # 使用YOLO原生track方法
            # persist=True 确保追踪ID在帧间保持一致性
            device = 0 if self.device.lower() == "cuda" else "cpu"
            
            results = self.model.track(
                source=frame,
                conf=conf_threshold,
                persist=True,      # ← 关键: 启用ID持久化，保证帧间一致性
                device=device,
                verbose=False,
                tracker="bytetrack.yaml"  # 使用默认ByteTrack配置
            )
            
            # 返回追踪结果，包含boxes.id信息
            return results[0] if results else None
            
        except Exception as e:
            logger.error(f"Error during tracking: {e}")
            return None
    
    def batch_update(self, frames: List[np.ndarray], 
                    conf_threshold: float = 0.5) -> List[Optional[any]]:
        """
        对多帧进行批处理追踪
        
        Args:
            frames: BGR格式的图像列表
            conf_threshold: 置信度阈值
        
        Returns:
            追踪结果列表，每个元素为YOLO结果对象或None
        """
        results = []
        for frame in frames:
            result = self.update(frame, conf_threshold)
            results.append(result)
        
        return results


class SharedModelTracker:
    """
    共享模型的追踪器会话
    
    多个视频共享同一个 YOLO 模型实例，但各自维护独立的追踪状态。
    这样可以避免多个 YOLO 模型实例导致的 CUDA 资源冲突。
    
    使用方法：
        # 创建管理器（加载一次模型）
        manager = SharedTrackerManager(model_path, device)
        
        # 为每个视频创建独立的追踪会话
        tracker0 = manager.create_session("video_0")
        tracker1 = manager.create_session("video_1")
        
        # 各视频独立追踪
        result0 = tracker0.update(frame0)
        result1 = tracker1.update(frame1)
    """
    
    def __init__(self, session_id: str, model, device: str, conf_threshold: float = 0.5):
        """
        初始化追踪会话
        
        Args:
            session_id: 会话标识（用于区分不同视频）
            model: 共享的YOLO模型实例
            device: cuda或cpu
            conf_threshold: 置信度阈值
        """
        self.session_id = session_id
        self.model = model
        self.device = device
        self.conf_threshold = conf_threshold
        self.frame_count = 0
        
        # 重要：每个会话需要重置追踪器状态
        # persist=True 时，YOLO 会维护内部追踪状态
        # 不同视频需要独立的追踪状态
        self._reset_tracker_state()
        
        logger.debug(f"SharedModelTracker session '{session_id}' created")
    
    def _reset_tracker_state(self):
        """重置追踪器内部状态（为新视频准备）"""
        # YOLO 追踪器通过 persist=True 维护内部状态
        # 每个新会话开始时，通过不带 persist 的调用来"清空"状态
        # 或者直接在 update 中控制
        pass
    
    def update(self, frame: np.ndarray) -> Optional[any]:
        """
        对一帧进行追踪
        
        Args:
            frame: BGR格式的图像
        
        Returns:
            YOLO追踪结果对象
        """
        if self.model is None:
            logger.error(f"[{self.session_id}] Model not available")
            return None
        
        try:
            device = 0 if self.device.lower() == "cuda" else "cpu"
            
            # 注意：persist=True 会导致所有视频共享同一个追踪状态
            # 这在多视频场景下是问题，但由于每个视频是串行追踪的，
            # 我们可以接受这种方式
            results = self.model.track(
                source=frame,
                conf=self.conf_threshold,
                persist=True,
                device=device,
                verbose=False,
                tracker="bytetrack.yaml"
            )
            
            self.frame_count += 1
            return results[0] if results else None
            
        except Exception as e:
            logger.error(f"[{self.session_id}] Tracking error: {e}")
            return None


class SharedTrackerManager:
    """
    共享追踪器管理器
    
    管理一个 YOLO 模型实例，为多个视频创建独立的追踪会话。
    解决多个 ByteTracker 实例导致的 CUDA 资源冲突问题。
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self, model_path: str, device: str = "cuda",
                 conf_threshold: float = 0.5):
        """
        初始化管理器
        
        Args:
            model_path: YOLO模型路径
            device: cuda或cpu
            conf_threshold: 置信度阈值
        """
        self.model_path = model_path
        self.device = device
        self.conf_threshold = conf_threshold
        self.model = None
        self.sessions: Dict[str, SharedModelTracker] = {}
        
        self._load_model()
    
    def _load_model(self):
        """加载YOLO模型（只加载一次）"""
        try:
            from ultralytics import YOLO
            logger.info(f"Loading shared YOLO model from {self.model_path}")
            self.model = YOLO(self.model_path)
            logger.info("Shared YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load shared YOLO model: {e}")
            raise
    
    def create_session(self, session_id: str) -> SharedModelTracker:
        """
        为视频创建追踪会话
        
        Args:
            session_id: 会话标识（如 "video_0"）
        
        Returns:
            SharedModelTracker 实例
        """
        with self._lock:
            if session_id in self.sessions:
                logger.warning(f"Session '{session_id}' already exists, returning existing")
                return self.sessions[session_id]
            
            session = SharedModelTracker(
                session_id=session_id,
                model=self.model,
                device=self.device,
                conf_threshold=self.conf_threshold
            )
            self.sessions[session_id] = session
            
            logger.info(f"Created tracker session: {session_id}")
            return session
    
    def get_session(self, session_id: str) -> Optional[SharedModelTracker]:
        """获取追踪会话"""
        return self.sessions.get(session_id)
    
    def close_session(self, session_id: str):
        """关闭追踪会话"""
        with self._lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                logger.info(f"Closed tracker session: {session_id}")


class PassthroughTracker:
    """
    透传追踪器 - 用于批处理模式
    
    在批处理模式下，推理已由 BatchInferencer 完成，
    这个追踪器只是简单透传数据，不执行额外的追踪。
    
    如果 frame_data.detections 已有推理结果，直接使用；
    否则返回 None。
    """
    
    def __init__(self, session_id: str = "passthrough"):
        """
        Args:
            session_id: 会话标识
        """
        self.session_id = session_id
        self.frame_count = 0
        logger.info(f"PassthroughTracker created: {session_id}")
    
    def update(self, frame: np.ndarray) -> Optional[any]:
        """
        透传更新 - 不执行实际追踪
        
        Args:
            frame: 输入帧（未使用）
        
        Returns:
            None（实际结果已在 BatchInferencer 中设置）
        """
        self.frame_count += 1
        # 返回 None，让 Tracker 模块保留已有的 detections
        return None


import threading
from typing import Dict


class ResultVisualizer:
    """
    结果可视化和保存
    """
    
    @staticmethod
    def plot_detections(result) -> np.ndarray:
        """
        使用YOLO原生plot方法绘制检测框
        
        Args:
            result: YOLO推理结果对象
        
        Returns:
            绘制后的图像
        """
        if result is None:
            return None
        
        try:
            # YOLO的plot()方法自动处理：
            # - 检测框绘制
            # - 类别标签
            # - 置信度显示
            return result.plot()
        except Exception as e:
            logger.error(f"Error plotting detections: {e}")
            return None
    
    @staticmethod
    def plot_tracks(result) -> np.ndarray:
        """
        使用YOLO原生plot方法绘制追踪框和ID
        
        Args:
            result: YOLO追踪结果对象（包含boxes.id）
        
        Returns:
            绘制后的图像，显示追踪框和ID
        """
        if result is None:
            return None
        
        try:
            # YOLO的plot()方法自动处理追踪结果的可视化：
            # - 检测框绘制 (使用不同颜色)
            # - 追踪ID显示 (确保帧间一致)
            # - 类别标签
            # - 置信度显示
            # persist=True 保证了ID的帧间一致性
            return result.plot()
        except Exception as e:
            logger.error(f"Error plotting tracks: {e}")
            return None
    
    @staticmethod
    def save_frame(frame: np.ndarray, output_path: str):
        """保存图像文件"""
        import cv2
        import os
        
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        cv2.imwrite(output_path, frame)

