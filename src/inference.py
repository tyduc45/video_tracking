"""
YOLO推理模块
"""

import logging
import numpy as np
import json
import os
from typing import List, Optional, Dict, Any
import time

logger = logging.getLogger(__name__)


class YOLOInferencer:
    """
    YOLO推理器 - 支持批处理和Engine加速

    动态模型管理：
    - 优先加载 .engine 文件
    - 从 meta 文件读取配置
    - 当 batch size 不匹配时自动重新导出 engine
    """

    META_FILE = "yolo12n_batch.meta"

    def __init__(self, model_path: str, model_dir: str = "../model",
                 device: str = "cuda", use_half: bool = True,
                 confidence_threshold: float = 0.5,
                 iou_threshold: float = 0.45,
                 batch_size: int = 16):
        self.model_dir = model_dir
        self.device = device
        self.use_half = use_half
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.requested_batch_size = batch_size

        self.model = None
        self.current_model_path = None
        self.current_batch_size = None

        self.meta = self._load_meta_config()
        self._smart_load_model(model_path)

    def _get_meta_path(self) -> str:
        return os.path.join(self.model_dir, self.META_FILE)

    def _load_meta_config(self) -> Dict[str, Any]:
        meta_path = self._get_meta_path()
        default_meta = {
            "model_name": "yolo12n",
            "batch_size": 16,
            "use_half": True,
            "input_size": [640, 640],
            "framework": "yolov8",
            "engine_batch_size": None,
            "last_request_batch_size": None,
        }

        try:
            if os.path.exists(meta_path):
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                    logger.info(f"Loaded meta config: {meta}")
                    for key, value in default_meta.items():
                        if key not in meta:
                            meta[key] = value
                    return meta
        except Exception as e:
            logger.warning(f"Failed to load meta config: {e}")

        return default_meta

    def _save_meta_config(self):
        meta_path = self._get_meta_path()
        try:
            os.makedirs(os.path.dirname(meta_path) or '.', exist_ok=True)
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(self.meta, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved meta config: {self.meta}")
        except Exception as e:
            logger.warning(f"Failed to save meta config: {e}")

    def _smart_load_model(self, model_path: str):
        """智能加载模型"""
        model_ext = os.path.splitext(model_path)[1].lower()
        base_name = os.path.splitext(os.path.basename(model_path))[0]

        pt_path = os.path.join(self.model_dir, f"{base_name}.pt")
        engine_path = os.path.join(self.model_dir, f"{base_name}.engine")

        engine_batch_size = self.meta.get("engine_batch_size")
        last_request = self.meta.get("last_request_batch_size")

        logger.info(f"Smart model loading:")
        logger.info(f"  Requested batch_size: {self.requested_batch_size}")
        logger.info(f"  Engine batch_size (meta): {engine_batch_size}")
        logger.info(f"  Last request batch_size: {last_request}")

        self.meta["last_request_batch_size"] = self.requested_batch_size

        use_engine = False

        if os.path.exists(engine_path):
            if engine_batch_size is not None and engine_batch_size >= self.requested_batch_size:
                logger.info(f"Engine exists with sufficient batch_size ({engine_batch_size} >= {self.requested_batch_size})")
                use_engine = True
            else:
                logger.info(f"Engine batch_size insufficient, need re-export")
                if os.path.exists(pt_path):
                    use_engine = self._export_engine(pt_path, engine_path, self.requested_batch_size)
        else:
            logger.info(f"Engine not found, attempting export")
            if os.path.exists(pt_path):
                use_engine = self._export_engine(pt_path, engine_path, self.requested_batch_size)

        if use_engine and os.path.exists(engine_path):
            self._load_model(engine_path)
            self.current_batch_size = self.meta.get("engine_batch_size", self.requested_batch_size)
        elif os.path.exists(pt_path):
            logger.info("Falling back to .pt model (dynamic batch size)")
            self._load_model(pt_path)
            self.current_batch_size = self.requested_batch_size
        else:
            self._load_model(model_path)
            self.current_batch_size = self.requested_batch_size

        self._save_meta_config()

    def _export_engine(self, pt_path: str, engine_path: str, batch_size: int) -> bool:
        """从 .pt 导出 .engine 文件"""
        try:
            from ultralytics import YOLO

            logger.info(f"Exporting TensorRT engine...")
            logger.info(f"  Source: {pt_path}")
            logger.info(f"  Target: {engine_path}")
            logger.info(f"  Batch size: {batch_size}")

            model = YOLO(pt_path)

            export_path = model.export(
                format='engine',
                half=self.use_half,
                batch=batch_size,
                device=0 if self.device.lower() == "cuda" else "cpu",
                simplify=True,
                workspace=4,
            )

            logger.info(f"Engine exported successfully: {export_path}")

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
        """填充 batch 到 engine 要求的大小"""
        if self.current_batch_size is None:
            return frames

        actual_size = len(frames)
        target_size = self.current_batch_size

        if actual_size >= target_size:
            return frames[:target_size]

        padding = [frames[-1]] * (target_size - actual_size)
        return frames + padding

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

        actual_frame_count = len(frames)
        model_ext = os.path.splitext(self.current_model_path or "")[1].lower()

        try:
            device = 0 if self.device.lower() == "cuda" else "cpu"

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
        """单帧推理（兼容接口）"""
        results = self.infer_batch([frame])
        return results[0] if results else []

    def __del__(self):
        self.model = None


class PassthroughTracker:
    """
    透传追踪器 - 用于批处理模式

    在批处理模式下，推理已由 BatchProcessor 完成，
    这个追踪器只是简单透传数据，不执行额外的追踪。
    """

    def __init__(self, session_id: str = "passthrough"):
        self.session_id = session_id
        self.frame_count = 0
        logger.info(f"PassthroughTracker created: {session_id}")

    def update(self, frame: np.ndarray) -> Optional[any]:
        """透传更新 - 不执行实际追踪"""
        self.frame_count += 1
        return None
