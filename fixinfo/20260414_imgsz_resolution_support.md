# 推理输入分辨率（imgsz）参数化支持

**日期**: 2026-04-14
**分支**: feature/parallel
**状态**: 完成

---

## 背景

原系统推理输入分辨率硬编码为 640×640（YOLO 默认值），无法通过配置或命令行调整。在高分辨率视频场景下，目标过小导致漏检，需要支持更高分辨率（如 1280×1280）的推理输入。

---

## 修改概览

| 文件 | 修改内容 |
|------|---------|
| `src/inference.py` | engine 命名加分辨率后缀；`model.export()` / `model.predict()` 传入 `imgsz` |
| `src/config.py` | 添加 `imgsz` 字段；`to_dict()` 导出该字段 |
| `src/main.py` | 添加 `--imgsz` CLI 参数；三处 `YOLOInferencer` 创建均传入 `imgsz` |

---

## 详细修改

### 1. `src/inference.py`

#### 1.1 `_smart_load_model` — engine 文件命名规则更新

**修改前**:
```
命名规范: {model_name}b{batch_size}.engine
例如: yolo12nb4.engine
```

**修改后**:
```
640分辨率:    {model_name}b{batch_size}.engine           （与旧命名兼容）
非640分辨率: {model_name}b{batch_size}_imgsz{N}.engine  （自动加后缀）
```

实现代码（`inference.py` L100-L102）：
```python
imgsz_suffix = f"_imgsz{self.imgsz}" if self.imgsz != 640 else ""
engine_path = os.path.join(
    self.model_dir, f"{base_name}b{self.requested_batch_size}{imgsz_suffix}.engine"
)
```

不同 `imgsz` 的 engine 文件独立共存，互不覆盖，例如：
```
yolo12nb4.engine          ← 640 分辨率，batch=4
yolo12nb4_imgsz1280.engine ← 1280 分辨率，batch=4
```

#### 1.2 `_export_engine` — 导出时传入 `imgsz`

**修改前**:
```python
export_path = model.export(
    format='engine',
    half=self.use_half,
    batch=batch_size,
    device=...,
    simplify=True,
    workspace=4,
)
```

**修改后**:
```python
export_path = model.export(
    format='engine',
    half=self.use_half,
    batch=batch_size,
    imgsz=self.imgsz,      # 新增
    device=...,
    simplify=True,
    workspace=4,
)
```

同时将 `imgsz` 写入 meta 文件持久化：
```python
self.meta["imgsz"] = self.imgsz
```

#### 1.3 `infer_batch` — 推理时传入 `imgsz`

**修改前**:
```python
results = self.model.predict(
    source=frames_to_infer,
    conf=self.confidence_threshold,
    iou=self.iou_threshold,
    half=self.use_half,
    device=device,
    verbose=False
)
```

**修改后**:
```python
results = self.model.predict(
    source=frames_to_infer,
    conf=self.confidence_threshold,
    iou=self.iou_threshold,
    half=self.use_half,
    device=device,
    imgsz=self.imgsz,      # 新增
    verbose=False
)
```

---

### 2. `src/config.py`

**新增 `imgsz` 字段**（推理配置块）：
```python
# 推理配置
device: str = "cpu"
confidence_threshold: float = 0.25
iou_threshold: float = 0.8
batch_size: int = 32
imgsz: int = 640           # 新增，默认保持 640
```

**`to_dict()` 新增导出**：
```python
'batch_size': self.batch_size,
'imgsz': self.imgsz,       # 新增
'queue_size': self.queue_size,
```

---

### 3. `src/main.py`

**新增 CLI 参数**：
```python
parser.add_argument('--imgsz', type=int, default=640,
                   help='推理输入分辨率（默认640，可设为1280等）')
```

**应用到 config**（命令行参数覆盖块）：
```python
config.imgsz = args.imgsz
```

**三处 `YOLOInferencer` 创建均传入 `imgsz`**：

| 位置 | 策略 |
|------|------|
| `run_chaotic_mode` (策略1) | `imgsz=config.imgsz` |
| `run_independent_mode` (策略2) | `imgsz=config.imgsz` |
| `run_batch_mode` (策略3) | `imgsz=config.imgsz` |

---

## 使用方法

```bash
# 默认 640 分辨率（行为不变）
python main.py -i ../videos/video0.mp4 --device cuda --batch-size 4 --strategy 3

# 1280 分辨率（首次运行自动导出 yolo12nb4_imgsz1280.engine）
python main.py -i ../videos/video0.mp4 --device cuda --batch-size 4 --imgsz 1280 --strategy 3
```

---

## 注意事项

1. **engine 文件需重新导出**：修改 `imgsz` 后，旧 `.engine` 不兼容，首次运行会自动从 `.pt` 重新导出对应分辨率的 engine 文件。

2. **640 分辨率 engine 文件命名不变**：`yolo12nb4.engine`，保持向后兼容，不影响已有 engine 文件。

3. **显存需求随分辨率增大**：

   | 分辨率 | 相对显存 | 相对速度 | 适用场景 |
   |--------|---------|---------|---------|
   | 640    | 基准     | 最快     | 常规目标检测 |
   | 1280   | ~4x     | ~1/4    | 小目标、高分辨率视频 |

4. **`imgsz` 对 `.pt` 模型同样生效**：`model.predict(imgsz=...)` 会在推理前自动 resize 输入帧，无需手动缩放。

---

## 验证清单

- [x] `inference.py`：engine 文件名含 `_imgsz{N}` 后缀（非640时）
- [x] `inference.py`：`model.export()` 传入 `imgsz`
- [x] `inference.py`：`model.predict()` 传入 `imgsz`
- [x] `inference.py`：meta 文件记录 `imgsz`
- [x] `config.py`：`imgsz` 字段默认值 640
- [x] `config.py`：`to_dict()` 包含 `imgsz`
- [x] `main.py`：`--imgsz` CLI 参数
- [x] `main.py`：三种策略均传入 `imgsz`
- [x] 640 分辨率下行为与修改前完全一致（向后兼容）
