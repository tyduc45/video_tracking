# 架构革新：多视频批处理推理系统

**日期**: 2026-01-23  
**类型**: 架构革新  
**状态**: 已实现

## 问题背景

原有架构中，每个视频独立处理，推理模块未能充分利用GPU的批处理能力。当有多个视频源时，需要一个能够跨视频批处理的架构来提高GPU利用率。

## 新架构设计

```
┌─────────────────────────────────────────────────────────────────────┐
│  Video0                           Video1                    ...     │
│  ┌───────┐                       ┌───────┐                          │
│  │Reader0│                       │Reader1│                          │
│  └───┬───┘                       └───┬───┘                          │
│      │                               │                              │
│  Queue0_read                     Queue1_read                        │
└──────┼───────────────────────────────┼──────────────────────────────┘
       │                               │
       └───────────┬───────────────────┘
                   │
           ┌───────▼───────┐
           │BatchInferencer│  从每个视频队列连续取K帧
           │  (单线程)      │  K = batch_size / num_videos
           └───────┬───────┘
                   │
       ┌───────────┴───────────────────┐
       │                               │
   Queue0_infer                    Queue1_infer
       │                               │
   ┌───▼────┐                      ┌───▼────┐
   │Tracker0│                      │Tracker1│  每个视频独立追踪器
   └───┬────┘                      └───┬────┘
       │                               │
   Queue0_save                     Queue1_save
       │                               │
   ┌───▼───┐                       ┌───▼───┐
   │ Saver0│                       │ Saver1│
   └───────┘                       └───────┘
```

### 核心设计原则

1. **生产者-消费者模式**
   - 每个视频的Reader是生产者，向各自的队列写帧
   - BatchInferencer是消费者，从所有队列取帧

2. **批处理帧提取策略（连续提取）**
   - 从Video0连续取K帧
   - 从Video1连续取K帧
   - ...
   - 组成batch进行推理
   - K = batch_size / num_videos

3. **独立追踪器**
   - 每个视频有独立的ByteTracker实例
   - 保证追踪ID在各视频内独立

## 实现文件

### 新增文件

- `src/batch_inference_system.py` - 批处理系统核心模块
  - `BatchConfig` - 批处理配置
  - `BatchInferencer` - 批推理器
  - `MultiVideoBatchPipeline` - 多视频批处理流水线管理器

### 修改文件

- `src/main.py` - 添加双模式支持
  - `--batch-size` 参数（默认32）
  - `--use-traditional` 参数（强制使用传统模式）
  - `run_batch_mode()` - 批处理模式
  - `run_traditional_mode()` - 传统模式

- `src/pipeline_data.py` - 添加 `video_index` 字段

## 使用方法

### 批处理模式（多视频自动启用）

```bash
python main.py -i video1.mp4 video2.mp4 -m model/yolo12n.engine -d cuda --batch-size 32
```

### 强制传统模式

```bash
python main.py -i video1.mp4 video2.mp4 -m model/yolo12n.engine -d cuda --use-traditional
```

### 单视频（自动使用传统模式）

```bash
python main.py -i video1.mp4 -m model/yolo12n.engine -d cuda
```

## 性能优势

| 场景 | 传统模式 | 批处理模式 |
|------|----------|------------|
| 2视频 | 2x单帧推理 | 1x批推理(batch=32) |
| 4视频 | 4x单帧推理 | 1x批推理(batch=32) |
| GPU利用率 | 低 | 高 |

## 注意事项

1. 批处理模式下，每个视频必须有**独立的追踪器实例**
2. 批处理大小应根据GPU显存调整
3. 单视频时自动使用传统模式（无需批处理）
