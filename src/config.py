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
    model_dir: str = "../model"                          # 模型所在目录
    model_name: str = "yolo12n"                          # 模型名称（无扩展名）
    model_path: str = ""                                 # 完整路径（会自动组装）
    use_engine: bool = True                              # 优先使用engine文件
    use_half: bool = True                                # 使用半精度
    model_type: str = "yolov8"                           # yolov8, yolov5等
    
    # 推理配置
    device: str = "cpu"                                  # cuda或cpu（优先使用cuda）
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    inference_batch_size: int = 16                       # 批处理大小（16或32）
    
    # Pipeline配置
    queue_size: int = 10
    max_pipelines: int = 10
    
    # 追踪配置
    tracker_type: str = "bytetrack"  # bytetrack, deepsort等
    track_high_thresh: float = 0.6
    track_low_thresh: float = 0.1
    track_buffer: int = 30
    
    # 输入输出配置
    input_dir: str = "videos"
    output_dir: str = "../result"                        # 默认为项目上一级
    
    # 可视化配置
    save_frames: bool = True
    save_video: bool = True
    save_fps: float = 30.0
    draw_boxes: bool = True
    draw_ids: bool = True
    draw_confidence: bool = True
    
    # 日志配置
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    def __post_init__(self):
        """初始化后处理，组装model_path"""
        # 自动检测cuda可用性
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
            # 优先使用engine文件
            if self.use_engine:
                engine_path = os.path.join(self.model_dir, f"{self.model_name}.engine")
                if os.path.exists(engine_path):
                    self.model_path = engine_path
                    return
            
            # 否则使用pt文件
            pt_path = os.path.join(self.model_dir, f"{self.model_name}.pt")
            if os.path.exists(pt_path):
                self.model_path = pt_path
                return
            
            # 如果都不存在，使用onnx
            onnx_path = os.path.join(self.model_dir, f"{self.model_name}.onnx")
            if os.path.exists(onnx_path):
                self.model_path = onnx_path
                return
            
            # 默认设为pt路径（可能不存在）
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
            'use_engine': self.use_engine,
            'use_half': self.use_half,
            'model_type': self.model_type,
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold,
            'inference_batch_size': self.inference_batch_size,
            'queue_size': self.queue_size,
            'max_pipelines': self.max_pipelines,
            'tracker_type': self.tracker_type,
            'track_high_thresh': self.track_high_thresh,
            'track_low_thresh': self.track_low_thresh,
            'track_buffer': self.track_buffer,
            'input_dir': self.input_dir,
            'output_dir': self.output_dir,
            'save_frames': self.save_frames,
            'save_video': self.save_video,
            'save_fps': self.save_fps,
            'draw_boxes': self.draw_boxes,
            'draw_ids': self.draw_ids,
            'draw_confidence': self.draw_confidence,
            'log_level': self.log_level,
            'log_file': self.log_file,
        }
    
    def validate(self, check_input_dir: bool = True) -> bool:
        """验证配置合法性
        
        Args:
            check_input_dir: 是否检查input_dir存在（当通过-i指定视频时可设为False）
        """
        errors = []
        
        if check_input_dir and not os.path.exists(self.input_dir):
            errors.append(f"Input directory does not exist: {self.input_dir}")
        
        if self.confidence_threshold < 0 or self.confidence_threshold > 1:
            errors.append(f"confidence_threshold must be in [0, 1], got {self.confidence_threshold}")
        
        if self.iou_threshold < 0 or self.iou_threshold > 1:
            errors.append(f"iou_threshold must be in [0, 1], got {self.iou_threshold}")
        
        if self.inference_batch_size < 1:
            errors.append(f"inference_batch_size must be >= 1, got {self.inference_batch_size}")
        
        if self.max_pipelines < 1:
            errors.append(f"max_pipelines must be >= 1, got {self.max_pipelines}")
        
        if errors:
            for error in errors:
                print(f"Config validation error: {error}")
            return False
        
        return True


def load_config(config_file: Optional[str] = None) -> Config:
    """
    加载配置
    
    Args:
        config_file: 配置文件路径，若为None则使用默认配置
    
    Returns:
        Config对象
    """
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
