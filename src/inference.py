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
    ByteTrack追踪器的包装类
    """
    
    def __init__(self, track_high_thresh: float = 0.6,
                 track_low_thresh: float = 0.1,
                 track_buffer: int = 30,
                 frame_rate: float = 30.0):
        """
        Args:
            track_high_thresh: 高阈值
            track_low_thresh: 低阈值
            track_buffer: 追踪缓冲区大小
            frame_rate: 视频帧率
        """
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.track_buffer = track_buffer
        self.frame_rate = frame_rate
        self.tracker = None
        
        self._load_tracker()
    
    def _load_tracker(self):
        """加载ByteTrack追踪器"""
        try:
            # 这是一个简化的实现
            # 实际使用时需要安装proper ByteTrack包
            logger.info("ByteTrack tracker initialized")
            
            # 简单的ID管理器
            self.next_id = 1
            self.tracks = {}  # id -> track_info
            
        except Exception as e:
            logger.error(f"Failed to load ByteTrack: {e}")
            raise
    
    def update(self, detections: List[dict], frame_id: int = None) -> List[dict]:
        """
        更新追踪结果
        
        Args:
            detections: 检测结果列表
            frame_id: 帧ID（可选）
        
        Returns:
            追踪结果列表，每个元素为:
            {
                'track_id': int,
                'class_id': int,
                'class_name': str,
                'confidence': float,
                'bbox': [x1, y1, x2, y2],
                'state': str,  # 'tracked' or 'lost'
            }
        """
        if not detections:
            return []
        
        try:
            # 这是一个简化的追踪实现
            # 实际使用应该集成真正的ByteTrack算法
            tracks = []
            
            for detection in detections:
                # 简单的ID分配（在实际应用中应使用匹配算法）
                track_id = self.next_id
                self.next_id += 1
                
                tracks.append({
                    'track_id': track_id,
                    'class_id': detection['class_id'],
                    'class_name': detection['class_name'],
                    'confidence': detection['confidence'],
                    'bbox': detection['bbox'],
                    'state': 'tracked',
                })
            
            return tracks
        
        except Exception as e:
            logger.error(f"Error during tracking: {e}")
            return []


class ResultVisualizer:
    """
    结果可视化和保存
    """
    
    @staticmethod
    @staticmethod
    def draw_detections(frame: np.ndarray, detections: List[dict]) -> np.ndarray:
        """
        在图像上绘制检测框
        
        Args:
            frame: 原始图像
            detections: 检测结果列表
        
        Returns:
            绘制后的图像
        """
        import cv2
        
        result = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # 绘制框 - 使用蓝色和较细的线条（更美观的样式）
            cv2.rectangle(result, (x1, y1), (x2, y2), (255, 0, 0), 1)
            
            # 绘制半透明背景的标签（更易读）
            label = f"{class_name}: {confidence:.2f}"
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # 绘制标签背景
            cv2.rectangle(result, 
                         (x1, y1 - label_size[1] - baseline - 4),
                         (x1 + label_size[0], y1),
                         (255, 0, 0), -1)
            
            # 绘制标签文本
            cv2.putText(result, label, (x1, y1 - baseline - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result
    
    @staticmethod
    def draw_tracks(frame: np.ndarray, tracks: List[dict]) -> np.ndarray:
        """
        在图像上绘制追踪框和ID
        
        Args:
            frame: 原始图像
            tracks: 追踪结果列表
        
        Returns:
            绘制后的图像
        """
        import cv2
        
        result = frame.copy()
        
        for track in tracks:
            x1, y1, x2, y2 = track['bbox']
            track_id = track['track_id']
            class_name = track['class_name']
            confidence = track['confidence']
            
            # 根据状态选择颜色 - 使用蓝色和其他清晰的颜色
            if track['state'] == 'tracked':
                color = (255, 0, 0)  # 蓝色 - 已追踪的
            elif track['state'] == 'confirmed':
                color = (0, 255, 255)  # 青色 - 已确认的
            else:
                color = (0, 0, 255)  # 红色 - 临时的
            
            # 绘制框 - 使用更细的线条和清晰的颜色（1像素宽度）
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 1)
            
            # 绘制追踪ID标签（更醒目）
            label = f"ID:{track_id}"
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # 绘制ID背景（半透明效果）
            cv2.rectangle(result,
                         (x1, y1 - label_size[1] - baseline - 6),
                         (x1 + label_size[0] + 4, y1 - 2),
                         color, -1)
            
            # 绘制ID文本（白色，易读）
            cv2.putText(result, label, (x1 + 2, y1 - baseline - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 绘制类别和置信度标签
            detail_label = f"{class_name}: {confidence:.2f}"
            detail_size, detail_baseline = cv2.getTextSize(detail_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            
            cv2.rectangle(result,
                         (x1, y2),
                         (x1 + detail_size[0], y2 + detail_size[1] + 4),
                         color, -1)
            
            cv2.putText(result, detail_label, (x1 + 2, y2 + detail_size[1] + 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return result
    
    @staticmethod
    def save_frame(frame: np.ndarray, output_path: str):
        """保存图像文件"""
        import cv2
        import os
        
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        cv2.imwrite(output_path, frame)
