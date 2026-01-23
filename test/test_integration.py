"""
集成测试
"""

import pytest
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import Config
from inference import YOLOInferencer
from pipeline_manager import PipelineManager
from video_source import LocalVideoSource


class TestPipelineCreation:
    """管道创建集成测试"""
    
    def test_pipeline_manager_creation(self):
        """测试管道管理器创建"""
        manager = PipelineManager(output_dir="result/test")
        
        assert manager is not None
        assert manager.output_dir == "result/test"
    
    def test_pipeline_manager_status(self):
        """测试管道管理器状态"""
        manager = PipelineManager(output_dir="result/test")
        
        # 初始时应该没有管道
        status = manager.get_status()
        assert isinstance(status, dict)


class TestVideoSourceCreation:
    """视频源创建测试"""
    
    def test_local_video_source_creation(self):
        """测试本地视频源创建"""
        try:
            source = LocalVideoSource("../videos/video0.mp4")
            assert source is not None
        except FileNotFoundError:
            pytest.skip("Test video not found")
    
    def test_video_source_metadata(self):
        """测试视频源元数据"""
        try:
            source = LocalVideoSource("../videos/video0.mp4")
            
            fps = source.get_fps()
            frame_count = source.get_frame_count()
            resolution = source.get_resolution()
            
            assert fps > 0
            assert frame_count > 0
            assert resolution is not None
            
            source.close()
        except FileNotFoundError:
            pytest.skip("Test video not found")


class TestConfigIntegration:
    """配置集成测试"""
    
    def test_config_with_custom_paths(self):
        """测试自定义路径配置"""
        config = Config(
            model_dir="../model",
            output_dir="../result",
        )
        
        assert config.model_dir == "../model"
        assert config.output_dir == "../result"
    
    def test_config_device_detection(self):
        """测试设备检测集成"""
        config = Config()
        
        # 应该正确检测设备
        assert config.device in ["cpu", "cuda"]
    
    def test_full_config_initialization(self):
        """测试完整配置初始化"""
        config = Config(
            model_dir="../model",
            model_name="yolo12n",
            device="cpu",
            use_engine=False,
            use_half=False,
            inference_batch_size=16,
        )
        
        # 所有参数应该正确设置
        assert config.model_dir == "../model"
        assert config.model_name == "yolo12n"
        assert config.device == "cpu"
        assert config.use_engine is False
        assert config.use_half is False
        assert config.inference_batch_size == 16


class TestInferencePipeline:
    """推理管道集成测试"""
    
    def test_inference_with_config(self):
        """测试使用Config的推理"""
        config = Config(device="cpu", use_engine=False, use_half=False)
        
        try:
            inferencer = YOLOInferencer(
                model_path=config.model_path,
                model_dir=config.model_dir,
                device=config.device,
                use_half=config.use_half,
            )
            
            # 创建示例帧
            frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            # 推理
            result = inferencer.infer(frame)
            assert result is not None
            
        except FileNotFoundError:
            pytest.skip("Model file not found")
    
    def test_batch_inference_pipeline(self):
        """测试批处理推理管道"""
        config = Config(device="cpu", use_engine=False, use_half=False)
        
        try:
            inferencer = YOLOInferencer(
                model_path=config.model_path,
                model_dir=config.model_dir,
                device=config.device,
                batch_size=8,
            )
            
            # 创建示例帧
            frames = [
                np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                for _ in range(8)
            ]
            
            # 批推理
            results = inferencer.infer_batch(frames)
            assert len(results) == len(frames)
            
        except FileNotFoundError:
            pytest.skip("Model file not found")


class TestEndToEnd:
    """端到端测试"""
    
    def test_full_workflow(self):
        """测试完整工作流"""
        # 1. 配置
        config = Config(device="cpu")
        
        try:
            # 2. 创建推理器
            inferencer = YOLOInferencer(
                model_path=config.model_path,
                model_dir=config.model_dir,
                device=config.device,
            )
            
            # 3. 创建管道管理器
            manager = PipelineManager(output_dir="result/e2e_test")
            
            # 4. 生成示例数据
            frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            # 5. 推理
            result = inferencer.infer(frame)
            
            # 6. 验证
            assert result is not None
            assert manager is not None
            
        except FileNotFoundError:
            pytest.skip("Model file not found")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
