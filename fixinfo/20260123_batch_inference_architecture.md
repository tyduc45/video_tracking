# 批处理推理架构革新

**日期**: 2026-01-23  
**状态**: ✅ 已完成

## 概述

实现多视频批处理推理系统，将旧方法（单视频流水线）和新方法（批处理推理）结合，提高多视频场景下的推理效率。

## 架构设计

```
Reader0 ──┐                              ┌──> Tracker0 (PassThrough) ──> Saver0
Reader1 ──┼──> BatchProcessor ──────────┼──> Tracker1 (PassThrough) ──> Saver1
Reader2 ──┤   (GPU批推理)                ├──> Tracker2 (PassThrough) ──> Saver2
...      ─┘                              └──> ...
```

### 核心组件

| 组件 | 功能 |
|------|------|
| `BatchCollector` | 按 k 值从各视频队列收集帧 |
| `ResultDistributor` | 使用 head/offset 机制分发推理结果 |
| `MultiVideoBatchProcessor` | 整合：收集 → 推理 → 分发 |
| `MultiVideoPipeline` | 流水线管理 |

## k 值算法

每个视频可以打进批次的帧数量：

```python
def calculate_k_values(num_videos: int, batch_size: int = 32) -> List[int]:
    base_k = batch_size // num_videos  # 向下取整
    remainder = batch_size % num_videos
    k_values = [base_k] * (num_videos - 1)
    k_values.append(base_k + remainder)  # 最后一个补齐
    return k_values
```

**示例**: n=15, batch_size=32
- base_k = 32 // 15 = 2
- k_values = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,4]

## 批次打包顺序

```
batch = [cv0f0, cv0f1, ..., cv0f(k0-1), cv1f0, cv1f1, ..., cv1f(k1-1), ...]
```

## 分发机制 (head/offset)

```python
head = 0
for video_idx in range(n):
    offset = actual_k_values[video_idx]
    video_frames = buffer[head : head + offset]
    result_dict[video_idx].extend(video_frames)
    head += offset
```

## 修改的文件

### 1. `src/batch_inference_system.py`
- 完全重写，实现新的批处理架构
- 添加 `calculate_k_values()` 函数
- 添加 `FrameMeta` 数据类
- 添加 `BatchCollector` 类
- 添加 `ResultDistributor` 类
- 添加 `MultiVideoBatchProcessor` 类
- 修改 `MultiVideoPipeline` 类

### 2. `src/inference.py`
- 重写 `YOLOInferencer` 类，支持动态模型管理
- 从 `yolo12n_batch.meta` 读取配置
- 自动检测 batch size 是否匹配
- 支持自动导出 TensorRT engine
- 添加 batch 填充功能（用于固定 batch size 的 engine）

### 3. `src/main.py`
- 添加信号处理器（Ctrl+C 支持）
- 修改 `run_batch_mode()` 使用 `PassthroughTracker`
- 设置全局停止事件

### 4. `src/pipeline_modules.py`
- `Reader.run()`: 移除单视频结束时的全局 `stop_event.set()`
- 防止单个视频结束导致所有视频停止

### 5. `src/visualizer.py`
- 修改 `process_frame()` 支持两种 detections 格式
- 添加 `_draw_detections()` 方法
- 支持 YOLO 结果对象（有 `.plot()` 方法）
- 支持 `List[dict]` 格式（批处理模式）

### 6. `model/yolo12n_batch.meta`
- 添加新字段：`engine_batch_size`, `last_request_batch_size`, `engine_export_time`

## 解决的问题

### 1. CUDA/TensorRT 多实例冲突
**问题**: 多个 ByteTracker 同时加载 YOLO 模型导致 CUDA Error 700  
**解决**: 批处理模式使用 `PassthroughTracker`，只有一个 `YOLOInferencer` 加载模型

### 2. TensorRT batch size 不匹配
**问题**: `.engine` 文件固定 batch size，动态帧数导致推理错误  
**解决**: 
- 自动填充 batch 到固定大小
- 自动从 `.pt` 重新导出匹配 batch size 的 engine
- 回退到 `.pt` 模型（支持动态 batch）

### 3. 队列满导致帧丢失
**问题**: "Failed to put frame in queue" 警告  
**解决**: 
- 减少 `BatchCollector` 的 timeout (0.5s → 0.1s)
- 增加队列大小 (100 → 200)

### 4. Ctrl+C 无法终止程序
**问题**: 程序无响应中断信号  
**解决**: 
- 添加 `signal.SIGINT` 处理器
- 线程改为 `daemon=True`
- `stop()` 方法清空队列防止阻塞
- `wait()` 添加超时

### 5. visualizer 不支持 List[dict] 格式
**问题**: "'list' object has no attribute 'plot'"  
**解决**: `_draw_detections()` 方法同时支持 YOLO 对象和 dict 列表

## 使用方法

```bash
# 批处理模式（多视频）
python main.py -i video1.mp4 video2.mp4 -o result -d cuda --batch-size 32

# 传统模式（强制）
python main.py -i video1.mp4 video2.mp4 -o result -d cuda --use-traditional
```

## 性能数据

测试结果（2个视频，4062帧）：
- Total batches: 194
- Inference time: 24.18s
- 平均: ~168 FPS

## Meta 文件格式

```json
{
  "model_name": "yolo12n",
  "batch_size": 32,
  "use_half": true,
  "input_size": [640, 640],
  "framework": "yolov8",
  "engine_batch_size": 32,
  "last_request_batch_size": 32,
  "engine_export_time": "2026-01-23 17:45:00"
}
```
