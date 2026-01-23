"""
YOLO推理和ByteTrack追踪的具体实现
"""

import logging
import numpy as np
import json
import os
from typing import List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class YOLOInferencer:
    """
    YOLO推理器 - 支持批处理和Engine加速
    """
    
    def __init__(self, model_path: str, model_dir: str = "../model",
                 device: str = "cuda", use_half: bool = True,
                 confidence_threshold: float = 0.5,
                 iou_threshold: float = 0.45,
                 batch_size: int = 16):
        """
        Args:
            model_path: 完整的模型文件路径（.pt, .engine, .onnx）
            model_dir: 模型目录，用于读取meta文件
            device: cuda或cpu
            use_half: 是否使用半精度
            confidence_threshold: 置信度阈值
            iou_threshold: IOU阈值
            batch_size: 批处理大小（16或32）
        """
        self.model_path = model_path
        self.model_dir = model_dir
        self.device = device
        self.use_half = use_half
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.batch_size = batch_size
        
        self.model = None
        self.batch_buffer = []  # 缓存帧，达到batch_size时进行推理
        self._load_model()
        self._load_meta_config()
    
    def _load_meta_config(self):
        """加载meta配置文件"""
        try:
            meta_path = os.path.join(self.model_dir, "yolo12n_batch.meta")
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                    logger.info(f"Loaded meta config: {meta}")
                    # 可以从meta中读取batch_size等配置
                    if 'batch_size' in meta:
                        self.batch_size = meta['batch_size']
                    if 'use_half' in meta:
                        self.use_half = meta['use_half']
            else:
                logger.info(f"Meta config file not found: {meta_path}")
        except Exception as e:
            logger.warning(f"Failed to load meta config: {e}")
    
    def _load_model(self):
        """加载YOLO模型"""
        try:
            from ultralytics import YOLO
            
            logger.info(f"Loading YOLO model from {self.model_path}")
            logger.info(f"  Device: {self.device}")
            logger.info(f"  Use half precision: {self.use_half}")
            logger.info(f"  Batch size: {self.batch_size}")
            
            # 检测模型文件类型
            model_ext = os.path.splitext(self.model_path)[1].lower()
            logger.info(f"  Model format: {model_ext}")
            
            # 加载模型
            self.model = YOLO(self.model_path, task='detect')
            
            # 只有PT格式支持to()方法，其他格式（engine, onnx等）不支持
            if model_ext == '.pt':
                if self.device.lower() == "cuda":
                    self.model.to("cuda")
                else:
                    self.model.to("cpu")
                logger.info(f"YOLO PT model loaded and moved to {self.device}")
            else:
                # Engine和ONNX格式：不需要to()，device在predict时指定
                logger.info(f"YOLO {model_ext.upper()} model loaded (device will be specified in predict)")
        
        except ImportError:
            logger.error("ultralytics package not found. Install it with: pip install ultralytics")
            raise
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def infer_batch(self, frames: List[np.ndarray]) -> List[List[dict]]:
        """
        批量推理多帧
        
        Args:
            frames: BGR格式的图像列表
        
        Returns:
            检测结果列表，每个元素对应一帧的检测结果列表
        """
        if self.model is None or not frames:
            return [[] for _ in frames]
        
        try:
            # 确定设备参数
            device = 0 if self.device.lower() == "cuda" else "cpu"
            
            # 使用predict而不是直接调用model()
            results = self.model.predict(
                source=frames,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                half=self.use_half,
                device=device,
                verbose=False  # 减少日志输出
            )
            
            all_detections = []
            for result in results:
                detections = []
                for detection in result.boxes:
                    class_id = int(detection.cls[0])
                    class_name = result.names[class_id]
                    confidence = float(detection.conf[0])
                    bbox = detection.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                    
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
            return [[] for _ in frames]
    
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

