# ByteTracker 初始化修复 (2026-01-23)

## 问题描述

运行主程序时出错：
```
2026-01-23 14:20:46,932 - __main__ - ERROR - Failed to load tracker: ByteTracker.__init__() missing 1 required positional argument: 'model_path'
```

ByteTracker 类需要 `model_path` 参数来加载 YOLO 模型进行追踪，但初始化时没有传递此参数。

---

## 修复内容

### 1. src/main.py - 主程序追踪器初始化

**位置**: 第 173-183 行

**改动前**:
```python
tracker = ByteTracker(
    track_high_thresh=config.track_high_thresh,
    track_low_thresh=config.track_low_thresh,
    track_buffer=config.track_buffer,
    frame_rate=30.0
)
```

**改动后**:
```python
tracker = ByteTracker(
    model_path=config.model_path,        # ← 添加
    device=config.device,                # ← 添加
    track_high_thresh=config.track_high_thresh,
    track_low_thresh=config.track_low_thresh,
    track_buffer=config.track_buffer,
    frame_rate=30.0
)
```

**原因**: config 对象已包含 model_path 和 device 配置信息，直接使用即可

---

### 2. src/examples.py - 所有 6 个示例代码

**修复的示例**:
- example1_single_video() - 第 43 行
- example2_multi_video() - 第 87 行  
- example3_mixed_sources() - 第 132 行
- example4_custom_inference() - 第 201 行
- example5_camera_monitoring() - 第 246 行
- example6_config_based() - 第 305 行

**统一修改模式**:

```python
# 添加模型路径定义
model_path = "../model/yolo12n.engine"

# 修改初始化，传入 model_path
tracker = ByteTracker(model_path=model_path)

# 或带自定义参数
tracker = ByteTracker(model_path=model_path, track_high_thresh=0.7)
```

---

## 技术说明

### ByteTracker 类要求

ByteTracker 类的构造函数签名：
```python
def __init__(self, model_path: str, device: str = "cuda",
             track_high_thresh: float = 0.6,
             track_low_thresh: float = 0.1,
             track_buffer: int = 30,
             frame_rate: float = 30.0):
```

**必需参数**:
- `model_path` (str) - YOLO 模型文件路径 (可以是 .engine, .pt, .onnx)

**可选参数** (有默认值):
- `device` - 计算设备 (cuda 或 cpu)
- `track_high_thresh` - 追踪高阈值
- `track_low_thresh` - 追踪低阈值  
- `track_buffer` - 追踪缓冲区大小
- `frame_rate` - 视频帧率

### 模型路径来源

在 main.py 中，模型路径可从配置获得：
```python
config.model_path  # 例如: '../model/yolo12n.engine'
```

在 examples.py 中，使用相对路径指向模型目录：
```python
model_path = "../model/yolo12n.engine"  # Engine 格式 (推荐)
# 或
model_path = "../model/yolo12n.pt"      # PyTorch 格式
```

---

## 修改统计

| 文件 | 修改数 | 位置 |
|------|--------|------|
| src/main.py | 1 处 | 第 173-183 行 |
| src/examples.py | 6 处 | 第 43, 87, 132, 201, 246, 305 行 |
| **总计** | **7 处** | - |

---

## 验证方法

### 运行主程序
```bash
cd src
python main.py
```

**预期结果**:
```
2026-01-23 14:20:46,932 - __main__ - INFO - Loading tracker...
2026-01-23 14:20:46,933 - inference - INFO - YOLO tracker loaded from ../model/yolo12n.engine
```

不再出现 `ByteTracker.__init__() missing 1 required positional argument: 'model_path'` 错误

### 运行示例代码
```bash
cd src
python examples.py
# 选择要运行的示例
```

---

## 后续检查

- ✅ ByteTracker 所有初始化都包含 model_path 参数
- ✅ 模型路径均指向有效的模型文件 (../model/yolo12n.engine)
- ✅ main.py 从 config 对象中获取路径
- ✅ examples.py 使用硬编码的相对路径

