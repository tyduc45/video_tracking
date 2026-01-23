# 🧪 测试框架 - 完整文档

## 📋 目录结构

```
test/
├── __init__.py                 # 包初始化
├── conftest.py                 # Pytest 共享 fixture
├── test_config.py              # 配置模块测试
├── test_inference.py           # 推理模块测试
├── test_integration.py         # 集成测试
├── run_all_tests.py            # Python 一键测试脚本
├── run_tests.sh                # Linux/Mac 测试脚本
├── run_tests.bat               # Windows 测试脚本
└── README.md                   # 本文件
```

## 🚀 快速开始

### 方式1: Python 脚本（推荐）

```bash
cd e:\cpp_review\video_object_search
python test/run_all_tests.py
```

**特点**:
- ✅ 跨平台（Windows/Linux/Mac）
- ✅ 完整的依赖检查
- ✅ 自动生成报告
- ✅ 测试时间统计

### 方式2: Linux/Mac Shell 脚本

```bash
cd e:\cpp_review\video_object_search
bash test/run_tests.sh
```

### 方式3: Windows 批处理

```cmd
cd e:\cpp_review\video_object_search
test\run_tests.bat
```

### 方式4: 直接使用 pytest

```bash
# 运行所有测试
pytest test/ -v

# 运行特定测试文件
pytest test/test_inference.py -v

# 运行特定测试
pytest test/test_config.py::TestCUDADetection::test_cuda_detection -v

# 生成覆盖率报告
pytest test/ --cov=src --cov-report=html
```

## 📊 测试套件组成

### 1. Config 模块测试 (`test_config.py`)

**测试覆盖**:
- ✅ 默认配置
- ✅ 自定义配置
- ✅ CUDA自动检测
- ✅ CUDA fallback
- ✅ 模型路径解析
- ✅ 模型优先级搜索
- ✅ Meta配置加载
- ✅ 配置序列化

**测试类**:
```python
TestConfigBasic          # 基础配置测试
TestCUDADetection        # CUDA检测测试
TestModelPathResolution  # 模型路径解析
TestMetaConfigLoading    # Meta配置加载
TestConfigValidation     # 参数验证
TestEnvironmentDetection # 环境检测
```

**运行**:
```bash
pytest test/test_config.py -v
```

---

### 2. 推理模块测试 (`test_inference.py`)

**测试覆盖**:
- ✅ 模型加载（多格式）
- ✅ 单帧推理
- ✅ 批处理推理
- ✅ 缓冲推理
- ✅ Buffer flush
- ✅ 设备配置
- ✅ 半精度配置

**测试类**:
```python
TestModelLoading          # 模型加载
TestSingleFrameInference  # 单帧推理
TestBatchInference        # 批处理推理
TestBufferedInference     # 缓冲推理
TestInferenceConfiguration # 配置测试
```

**运行**:
```bash
pytest test/test_inference.py -v
```

---

### 3. 集成测试 (`test_integration.py`)

**测试覆盖**:
- ✅ 管道创建
- ✅ 视频源创建
- ✅ 完整配置流程
- ✅ 推理管道
- ✅ 批处理管道
- ✅ 端到端工作流

**测试类**:
```python
TestPipelineCreation     # 管道创建
TestVideoSourceCreation  # 视频源创建
TestConfigIntegration    # 配置集成
TestInferencePipeline    # 推理管道
TestEndToEnd             # 端到端测试
```

**运行**:
```bash
pytest test/test_integration.py -v
```

---

## 🔧 Pytest 配置 (`conftest.py`)

### 提供的 Fixtures

```python
@pytest.fixture
def temp_config():
    """临时测试配置"""
    return Config(device="cpu", use_engine=False, use_half=False)

@pytest.fixture
def sample_frame():
    """640x640x3 样本帧"""
    return np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

@pytest.fixture
def sample_frames(sample_frame):
    """16帧样本数据"""
    return [sample_frame.copy() for _ in range(16)]

@pytest.fixture(scope="session")
def project_root():
    """项目根目录"""
    return Path(__file__).parent.parent

@pytest.fixture(scope="session")
def model_dir(project_root):
    """模型目录"""
    return project_root / 'model'
```

### 使用示例

```python
def test_inference(sample_frame, temp_config):
    """在测试中使用 fixture"""
    config = temp_config
    inferencer = YOLOInferencer(
        model_path=config.model_path,
        device=config.device,
    )
    result = inferencer.infer(sample_frame)
    assert result is not None
```

## 📈 测试输出示例

### 成功运行

```
================================
开始测试运行 - 推理系统完整测试套件
================================

📦 检查依赖
✅ pytest              (测试框架)
✅ torch               (PyTorch)
✅ opencv              (OpenCV)
✅ ultralytics         (YOLO)
✅ numpy               (NumPy)

======================================================================
🧪 运行 Pytest 测试
======================================================================

test/test_config.py::TestConfigBasic::test_default_config PASSED
test/test_config.py::TestConfigBasic::test_custom_config PASSED
test/test_config.py::TestCUDADetection::test_cuda_detection PASSED
...
test/test_integration.py::TestEndToEnd::test_full_workflow PASSED

======================= 20 passed in 2.34s =======================

📊 测试总结
======================================================================
总体状态: ✅ 全部通过
时间戳: 20260123_142530
报告目录: test_reports/
```

## 📊 覆盖率报告

### 生成HTML覆盖率报告

```bash
pytest test/ --cov=src --cov-report=html
```

然后打开 `htmlcov/index.html`

### 查看终端覆盖率

```bash
pytest test/ --cov=src --cov-report=term-missing
```

输出示例:
```
Name                    Stmts   Miss  Cover   Missing
─────────────────────────────────────────────────────
src/__init__.py             1      0   100%
src/config.py              85      5    94%   42-46,120
src/inference.py          120      8    93%   87-92,145
─────────────────────────────────────────────────────
TOTAL                     206     13    93%
```

## 🔄 CI/CD 流水线

### GitHub Actions 工作流

**.github/workflows/test.yml** - 完整的测试流水线

**触发条件**:
- Push 到 main/develop 分支
- Pull Request 到 main/develop 分支

**运行内容**:
- 多Python版本测试 (3.8, 3.9, 3.10, 3.11)
- 多操作系统测试 (Ubuntu, Windows, macOS)
- 代码风格检查
- 安全扫描
- 覆盖率报告

### 查看CI/CD状态

```bash
# GitHub Actions
https://github.com/your-repo/video_object_search/actions
```

## ⚙️ 配置文件

### pytest.ini 推荐配置

在项目根目录创建 `pytest.ini`:

```ini
[pytest]
testpaths = test
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
markers =
    slow: 标记为慢速测试
    gpu: 需要GPU的测试
    integration: 集成测试
```

### tox.ini 多环境测试

```ini
[tox]
envlist = py38,py39,py310,py311

[testenv]
deps =
    pytest
    pytest-cov
    torch
    opencv-python
    ultralytics
    numpy
commands = pytest test/
```

运行:
```bash
pip install tox
tox
```

## 🐛 故障排查

### 问题1: "ModuleNotFoundError: No module named 'pytest'"

**解决**:
```bash
pip install pytest pytest-cov
```

### 问题2: 找不到模型文件

**解决**: 测试会跳过需要模型文件的测试
```
SKIPPED [2] test_inference.py:25: Model file not found
```

确保模型目录结构:
```
../model/
├── yolo12n.pt (或 .engine/.onnx)
└── yolo12n_batch.meta (可选)
```

### 问题3: CUDA 内存不足

**解决**:
```python
# 在 test 中使用 CPU
@pytest.fixture
def temp_config():
    return Config(device="cpu")
```

### 问题4: CI/CD 失败

**调试**:
```bash
# 本地重现 CI 环境
python -m venv venv_test
source venv_test/bin/activate  # Linux/Mac
# 或
venv_test\Scripts\activate  # Windows

pip install torch opencv-python ultralytics pytest
python test/run_all_tests.py
```

## 📝 编写新测试

### 基本模板

```python
import pytest
from src.config import Config
from src.inference import YOLOInferencer

class TestMyFeature:
    """我的特性测试"""
    
    def test_something(self, temp_config):
        """测试某个功能"""
        config = temp_config
        
        # 准备
        inferencer = YOLOInferencer(
            model_path=config.model_path,
            device=config.device,
        )
        
        # 执行
        result = inferencer.infer(frame)
        
        # 验证
        assert result is not None
        assert isinstance(result, list)
    
    def test_error_handling(self):
        """测试错误处理"""
        with pytest.raises(ValueError):
            Config(device="invalid")
    
    @pytest.mark.parametrize("batch_size", [1, 4, 8, 16])
    def test_batch_sizes(self, batch_size, temp_config):
        """参数化测试"""
        config = Config(inference_batch_size=batch_size)
        assert config.inference_batch_size == batch_size
```

### 运行自定义测试

```bash
pytest test/test_config.py::TestMyFeature::test_something -v
```

## 📚 扩展阅读

- [Pytest 官方文档](https://docs.pytest.org/)
- [GitHub Actions 文档](https://docs.github.com/en/actions)
- [测试驱动开发 (TDD)](https://en.wikipedia.org/wiki/Test-driven_development)

## ✅ 最佳实践

1. **每次提交前运行测试**
   ```bash
   python test/run_all_tests.py
   ```

2. **在CI/CD前运行本地测试**
   ```bash
   pytest test/ --tb=short
   ```

3. **定期检查覆盖率**
   ```bash
   pytest test/ --cov=src --cov-report=html
   ```

4. **为新功能添加测试**
   - 遵循 TDD 流程
   - 先写测试，后写代码

5. **保持测试独立**
   - 不依赖执行顺序
   - 使用 fixture 初始化
   - 使用 mocking 隔离外部依赖

---

**版本**: 1.0  
**更新**: 2026-01-23  
**维护**: 推理系统小组
