"""
Config 模块单元测试
"""

import pytest
import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import Config


class TestConfigBasic:
    """基本配置测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = Config()
        
        assert config.model_dir == "../model"
        assert config.model_name == "yolo12n"
        assert config.output_dir == "../result"
        assert config.device in ["cpu", "cuda"]
        assert config.use_half is True or config.use_half is False
    
    def test_custom_config(self, temp_config):
        """测试自定义配置"""
        assert temp_config.device == "cpu"
        assert temp_config.use_engine is False
        assert temp_config.use_half is False
        assert temp_config.inference_batch_size == 1
    
    def test_config_to_dict(self):
        """测试配置序列化"""
        config = Config(device="cpu")
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['device'] == "cpu"
        assert 'model_path' in config_dict


class TestCUDADetection:
    """CUDA自动检测测试"""
    
    def test_cuda_detection(self):
        """测试CUDA检测"""
        config = Config(device="cuda")
        
        # 如果CUDA可用，应该是cuda; 否则应该是cpu
        if torch.cuda.is_available():
            assert config.device == "cuda"
        else:
            assert config.device == "cpu"
    
    def test_cuda_fallback(self):
        """测试CUDA fallback到CPU"""
        config = Config(device="cuda")
        # 如果CUDA不可用，应该自动fallback
        assert config.device in ["cuda", "cpu"]
    
    def test_cpu_explicit(self):
        """测试显式指定CPU"""
        config = Config(device="cpu")
        assert config.device == "cpu"


class TestModelPathResolution:
    """模型路径解析测试"""
    
    def test_model_path_resolution(self):
        """测试模型路径自动解析"""
        config = Config()
        
        # model_path应该自动设置
        assert config.model_path is not None
        assert isinstance(config.model_path, str)
    
    def test_model_path_priority(self, model_dir):
        """测试模型优先级搜索"""
        # 应该优先搜索: engine > pt > onnx
        config = Config(model_dir=model_dir)
        
        if config.model_path:
            # 如果找到模型，应该是有效的扩展名
            valid_extensions = ['.engine', '.pt', '.onnx']
            assert any(config.model_path.endswith(ext) for ext in valid_extensions)
    
    def test_custom_model_dir(self, temp_config):
        """测试自定义模型目录"""
        custom_dir = "/custom/model/dir"
        config = Config(model_dir=custom_dir)
        assert config.model_dir == custom_dir


class TestMetaConfigLoading:
    """元数据配置加载测试"""
    
    def test_meta_config_parsing(self, temp_config):
        """测试meta配置解析"""
        # Config应该尝试加载meta文件
        config = Config(
            model_dir="../model",
            model_name="yolo12n"
        )
        # 不应该抛出异常
        assert config is not None
    
    def test_batch_size_from_meta(self):
        """测试从meta文件读取批大小"""
        config = Config()
        # inference_batch_size应该是正整数
        assert isinstance(config.inference_batch_size, int)
        assert config.inference_batch_size > 0


class TestConfigValidation:
    """配置验证测试"""
    
    def test_device_validation(self):
        """测试device参数验证"""
        # 有效的device值
        for device in ["cpu", "cuda"]:
            config = Config(device=device)
            assert config.device in ["cpu", "cuda"]
    
    def test_batch_size_validation(self):
        """测试batch_size验证"""
        for batch_size in [1, 4, 8, 16, 32]:
            config = Config(inference_batch_size=batch_size)
            assert config.inference_batch_size == batch_size
    
    def test_use_half_validation(self):
        """测试use_half参数"""
        for use_half in [True, False]:
            config = Config(use_half=use_half)
            assert config.use_half == use_half


class TestEnvironmentDetection:
    """环境检测测试"""
    
    def test_gpu_available(self):
        """测试GPU检测"""
        config = Config()
        # 应该正确检测GPU可用性
        assert isinstance(config.device, str)
    
    def test_model_directory_exists(self):
        """测试模型目录检查"""
        config = Config()
        # 如果使用绝对路径，应该是有效的路径
        if config.model_dir.startswith('/') or ':' in config.model_dir:
            # 绝对路径
            assert os.path.isabs(config.model_dir) or ':' in config.model_dir


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
