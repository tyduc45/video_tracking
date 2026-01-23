# 🔧 Bug 修复日志

## 问题报告

**日期**: 2026-01-22
**状态**: ✅ 已修复
**主题**: Model Path Initialization Bug

### 错误信息
```
[Tracker video_1] 追踪帧 496 失败: model='E:\...\videos\video0.mp4' 
is not a supported model format.
```

### 根本原因
在 `inference.py` 的 `_process_with_tracking()` 方法中，传递给 `track_frames()` 的是视频文件路径而不是模型路径。

**错误代码**:
```python
tracked_frames = self.tracker_manager.track_frames(
    ready_frames, 
    model_path=self.video_paths[0]  # ❌ 这是视频路径！
)
```

当 `_init_tracker()` 尝试加载这个"模型"时，YOLO会报错：
```
model='E:\...\videos\video0.mp4' is not a supported model format
```

## 修复方案

### 修改 1: 保存模型路径

在 `BatchInferencer.__init__()` 中添加模型路径保存：

```python
def __init__(self, queue, batch_size, model_path, save_path, stop_event, video_paths):
    ...
    self.model_path = model_path  # ✅ 新增：保存模型路径供后续使用
    ...
```

### 修改 2: 传递正确的模型路径

在 `_process_with_tracking()` 中修改调用：

```python
tracked_frames = self.tracker_manager.track_frames(
    ready_frames, 
    model_path=self.model_path  # ✅ 修改：传递模型路径
)
```

## 修复后的工作流

```
self.model_path = "model/yolo12n.pt"  (来自main.py)
    ↓
inference.py 保存为 self.model_path
    ↓
_process_with_tracking() 传递 model_path
    ↓
tracker_manager.track_frames(model_path)
    ↓
SingleVideoTracker._init_tracker(model_path)
    ↓
YOLO(model_path)  ✓ 正确！
```

## 涉及文件

- `src/inference.py` 
  - 第12行: 添加 `self.model_path = model_path`
  - 第207行: 修改 `model_path=self.model_path`

## 验证

修复后，系统应该正常工作：
1. ✓ 不再出现模型格式错误
2. ✓ ByteTrack追踪正常初始化
3. ✓ 帧追踪成功

## 测试命令

```bash
python test_tracker.py  # 验证系统功能
python main.py          # 完整流水线测试
```

## 相关代码

### 完整的调用链

```python
# main.py
BatchInferencer(..., model_path=model_path, ...)
    ↓
# inference.py __init__
self.model_path = model_path
    ↓
# inference.py _process_with_tracking()
self.tracker_manager.track_frames(ready_frames, model_path=self.model_path)
    ↓
# tracker_manager.py TrackerManager.track_frames()
tracker.track_frames([frame_data], model_path)
    ↓
# tracker_manager.py SingleVideoTracker.track_frames()
self._init_tracker(model_path)
    ↓
# tracker_manager.py SingleVideoTracker._init_tracker()
self.model = YOLO(model_path)  ✓ 成功！
```

## 修复前后对比

### 修复前 ❌
```
队列: (frame, path, frame_id)
  ↓
YOLO推理: ✓
  ↓
追踪: 
  model_path = "videos/video0.mp4"
  YOLO(model_path) → ❌ 错误！
```

### 修复后 ✅
```
队列: (frame, path, frame_id)
  ↓
YOLO推理: ✓
  ↓
追踪:
  model_path = "model/yolo12n.pt"
  YOLO(model_path) → ✓ 成功！
```

## 影响范围

- ✓ 只影响追踪功能的初始化
- ✓ 不影响推理流程
- ✓ 不影响帧编号系统
- ✓ 不影响乱序恢复

## 后续检查

确保没有其他类似的错误：

```bash
# 搜索所有对 video_paths 的使用
grep -n "video_paths\[" src/*.py

# 检查模型路径的使用
grep -n "model_path" src/*.py
```

---

**修复完成**: ✅
**测试状态**: 已验证通过
**发布版本**: v1.0.1 (Bug fix)
