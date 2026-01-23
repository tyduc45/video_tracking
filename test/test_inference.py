"""
推理模块单元测试
"""

import pytest
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import Config
from inference import YOLOInferencer


class TestModelLoading:
    """模型加载测试"""
    
    def test_model_loading_success(self, temp_config):
        """测试模型加载成功"""
        config = temp_config
        
        try:
            inferencer = YOLOInferencer(
                model_path=config.model_path,
                model_dir=config.model_dir,
                device=config.device,
                use_half=config.use_half,
                batch_size=config.inference_batch_size,
            )
            assert inferencer is not None
            assert inferencer.model is not None
        except FileNotFoundError:
            pytest.skip("Model file not found")
        except Exception as e:
            pytest.fail(f"Failed to load model: {e}")
    
    def test_model_loading_different_formats(self):
        """测试不同格式的模型加载"""
        config = Config(device="cpu", use_engine=False, use_half=False)
        
        try:
            # 这个测试依赖于实际的模型文件存在
            inferencer = YOLOInferencer(
                model_path=config.model_path,
                model_dir=config.model_dir,
                device=config.device,
            )
            assert inferencer.model is not None
        except FileNotFoundError:
            pytest.skip("Model file not found, skipping format test")


class TestSingleFrameInference:
    """单帧推理测试"""
    
    def test_single_frame_inference(self, sample_frame, temp_config):
        """测试单帧推理"""
        config = temp_config
        
        try:
            inferencer = YOLOInferencer(
                model_path=config.model_path,
                model_dir=config.model_dir,
                device=config.device,
                use_half=config.use_half,
            )
            
            # 推理
            result = inferencer.infer(sample_frame)
            
            # 验证结果
            assert result is not None
            assert isinstance(result, list)
            
        except FileNotFoundError:
            pytest.skip("Model file not found")
    
    def test_single_frame_shape(self, sample_frame, temp_config):
        """测试单帧shape验证"""
        config = temp_config
        
        # 确保输入是正确的shape
        assert sample_frame.shape == (640, 640, 3)
        assert sample_frame.dtype == np.uint8
    
    def test_inference_with_empty_frame(self, temp_config):
        """测试空白帧推理"""
        config = temp_config
        empty_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        
        try:
            inferencer = YOLOInferencer(
                model_path=config.model_path,
                model_dir=config.model_dir,
                device=config.device,
            )
            result = inferencer.infer(empty_frame)
            assert result is not None
        except FileNotFoundError:
            pytest.skip("Model file not found")


class TestBatchInference:
    """批处理推理测试"""
    
    def test_batch_inference(self, sample_frames, temp_config):
        """测试批处理推理"""
        config = temp_config
        
        try:
            inferencer = YOLOInferencer(
                model_path=config.model_path,
                model_dir=config.model_dir,
                device=config.device,
                use_half=config.use_half,
                batch_size=len(sample_frames),
            )
            
            # 批推理
            results = inferencer.infer_batch(sample_frames)
            
            # 验证结果
            assert results is not None
            assert isinstance(results, list)
            assert len(results) == len(sample_frames)
            
        except FileNotFoundError:
            pytest.skip("Model file not found")
    
    def test_batch_size_validation(self, sample_frames, temp_config):
        """测试批大小验证"""
        config = temp_config
        
        # 验证帧数和批大小
        assert len(sample_frames) == 16
        
        try:
            inferencer = YOLOInferencer(
                model_path=config.model_path,
                model_dir=config.model_dir,
                device=config.device,
                batch_size=16,
            )
            
            results = inferencer.infer_batch(sample_frames)
            assert len(results) == 16
            
        except FileNotFoundError:
            pytest.skip("Model file not found")


class TestBufferedInference:
    """缓冲推理测试"""
    
    def test_buffered_inference_accumulation(self, sample_frame, temp_config):
        """测试缓冲推理累积"""
        config = temp_config
        
        try:
            inferencer = YOLOInferencer(
                model_path=config.model_path,
                model_dir=config.model_dir,
                device=config.device,
                batch_size=4,
            )
            
            # 输入4帧，应该返回4个结果
            results_list = []
            for _ in range(4):
                result = inferencer.infer_with_buffering(sample_frame)
                if result:
                    results_list.append(result)
            
            # 缓冲区应该已满
            assert len(results_list) > 0
            
        except FileNotFoundError:
            pytest.skip("Model file not found")
    
    def test_buffer_flush(self, sample_frames, temp_config):
        """测试缓冲刷新"""
        config = temp_config
        
        try:
            inferencer = YOLOInferencer(
                model_path=config.model_path,
                model_dir=config.model_dir,
                device=config.device,
                batch_size=16,
            )
            
            # 输入10帧，然后刷新
            for frame in sample_frames[:10]:
                inferencer.infer_with_buffering(frame)
            
            # 刷新缓冲区
            results = inferencer.flush_buffer()
            assert results is not None
            
        except FileNotFoundError:
            pytest.skip("Model file not found")


class TestInferenceConfiguration:
    """推理配置测试"""
    
    def test_device_configuration(self, temp_config):
        """测试设备配置"""
        config = temp_config
        
        try:
            inferencer = YOLOInferencer(
                model_path=config.model_path,
                model_dir=config.model_dir,
                device=config.device,
            )
            
            assert inferencer.device == config.device
            
        except FileNotFoundError:
            pytest.skip("Model file not found")
    
    def test_half_precision_configuration(self):
        """测试半精度配置"""
        config = Config(device="cpu", use_half=False)
        
        try:
            inferencer = YOLOInferencer(
                model_path=config.model_path,
                model_dir=config.model_dir,
                device=config.device,
                use_half=config.use_half,
            )
            
            assert inferencer.use_half == config.use_half
            
        except FileNotFoundError:
            pytest.skip("Model file not found")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
