"""
Pytest 配置和共享 fixture
"""

import pytest
import sys
import os
import numpy as np

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import Config


@pytest.fixture
def temp_config():
    """临时配置对象"""
    return Config(
        device="cpu",
        use_engine=False,
        use_half=False,
        inference_batch_size=1,
    )


@pytest.fixture
def sample_frame():
    """生成示例帧 (640x640x3 BGR)"""
    return np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_frames(sample_frame):
    """生成示例帧批次"""
    return [sample_frame.copy() for _ in range(16)]


@pytest.fixture(scope="session")
def project_root():
    """获取项目根目录"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="session")
def model_dir(project_root):
    """获取模型目录"""
    return os.path.join(project_root, 'model')
