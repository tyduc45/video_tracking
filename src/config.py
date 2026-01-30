"""
配置管理模块
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """系统配置"""

    # 模型配置
    model_dir: str = "../model"
    model_name: str = "yolo12n"
    model_path: str = ""
    use_half: bool = True

    # 推理配置
    device: str = "cpu"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    batch_size: int = 32

    # Pipeline配置
    queue_size: int = 200

    # 输入输出配置
    input_dir: str = "videos"
    output_dir: str = "../result"

    # 可视化配置
    save_frames: bool = True
    save_video: bool = True
    save_fps: float = 30.0
    draw_boxes: bool = True
    draw_ids: bool = True
    draw_confidence: bool = True
    draw_trajectory: bool = True
    trajectory_length: int = 30
    trajectory_gap_timeout: float = 3.0  # 同一ID两次记录间隔超过此值（秒）则丢弃历史重新开始
    trajectory_expire_timeout: float = 3.0  # ID长时间无更新超过此值（秒）则丢弃整个轨迹
    trajectory_point_expire_timeout: float = 10.0  # 单个轨迹点超过此值（秒）后被删除

    # 日志配置
    log_level: str = "INFO"
    log_file: Optional[str] = None

    def __post_init__(self):
        """初始化后处理，组装model_path"""
        if self.device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning("CUDA is not available, falling back to CPU")
                    self.device = "cpu"
            except ImportError:
                pass

        if not self.model_path:
            engine_path = os.path.join(self.model_dir, f"{self.model_name}.engine")
            if os.path.exists(engine_path):
                self.model_path = engine_path
                return

            pt_path = os.path.join(self.model_dir, f"{self.model_name}.pt")
            if os.path.exists(pt_path):
                self.model_path = pt_path
                return

            onnx_path = os.path.join(self.model_dir, f"{self.model_name}.onnx")
            if os.path.exists(onnx_path):
                self.model_path = onnx_path
                return

            self.model_path = pt_path

    @classmethod
    def from_dict(cls, config_dict: dict):
        """从字典创建Config"""
        valid_fields = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered_dict)

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'model_dir': self.model_dir,
            'model_name': self.model_name,
            'model_path': self.model_path,
            'use_half': self.use_half,
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold,
            'batch_size': self.batch_size,
            'queue_size': self.queue_size,
            'input_dir': self.input_dir,
            'output_dir': self.output_dir,
            'save_frames': self.save_frames,
            'save_video': self.save_video,
            'save_fps': self.save_fps,
            'draw_boxes': self.draw_boxes,
            'draw_ids': self.draw_ids,
            'draw_confidence': self.draw_confidence,
            'draw_trajectory': self.draw_trajectory,
            'trajectory_length': self.trajectory_length,
            'trajectory_gap_timeout': self.trajectory_gap_timeout,
            'trajectory_expire_timeout': self.trajectory_expire_timeout,
            'trajectory_point_expire_timeout': self.trajectory_point_expire_timeout,
            'log_level': self.log_level,
            'log_file': self.log_file,
        }

    def validate(self, check_input_dir: bool = True) -> bool:
        """验证配置合法性"""
        errors = []

        if check_input_dir and not os.path.exists(self.input_dir):
            errors.append(f"Input directory does not exist: {self.input_dir}")

        if self.confidence_threshold < 0 or self.confidence_threshold > 1:
            errors.append(f"confidence_threshold must be in [0, 1], got {self.confidence_threshold}")

        if self.iou_threshold < 0 or self.iou_threshold > 1:
            errors.append(f"iou_threshold must be in [0, 1], got {self.iou_threshold}")

        if self.batch_size < 1:
            errors.append(f"batch_size must be >= 1, got {self.batch_size}")

        if errors:
            for error in errors:
                print(f"Config validation error: {error}")
            return False

        return True


def load_config(config_file: Optional[str] = None) -> Config:
    """加载配置"""
    if config_file and os.path.exists(config_file):
        import json
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        return Config.from_dict(config_dict)

    return Config()


def save_config(config: Config, config_file: str):
    """保存配置到文件"""
    import json
    os.makedirs(os.path.dirname(config_file) or '.', exist_ok=True)
    with open(config_file, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
