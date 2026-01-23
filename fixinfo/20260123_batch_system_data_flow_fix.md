# 批处理系统数据流修复

## 日期
2026-01-23

## 问题描述

用户运行批处理命令时遇到错误：
```bash
python main.py -i ../videos/video00.mp4 ../videos/video01.mp4 -m ../model/yolo12n.engine -d cuda --batch-size 32 --no-frames
```

### 问题1: AttributeError
```
AttributeError: 'NoneType' object has no attribute 'input_queue'
```
**原因**: `pipeline.tracker` 为 `None`，因为 `SingleVideoPipeline._create_modules()` 没有被调用。

### 问题2: 只生成一个视频
**原因**: 数据流没有正确从 `TrackerDispatcher` 流向 Saver。

### 问题3: Tracker线程冲突
**原因**: 批处理系统使用 `TrackerDispatcher` 进行追踪，但 `pipeline.start()` 仍然启动了单独的 Tracker 线程。

### 问题4: 共享追踪器实例
**原因**: 所有视频共享同一个 `tracker` 实例，导致追踪ID混乱。

## 修复方案

### 1. 修复 batch_inference_system.py

**TrackerDispatcher 添加 output_queues 参数**：
```python
def __init__(self, input_queues: List[Queue], tracker_instances: List,
             output_queues: List[Queue], num_videos: int, 
             stop_event: Optional[threading.Event] = None):
```

**追踪完成后发送到 Saver 队列**：
```python
# 发送到Saver队列
try:
    self.output_queues[video_idx].put(frame_data, timeout=2.0)
except Exception as e:
    logger.warning(f"Error sending to saver queue {video_idx}: {e}")
```

### 2. 修复 multi_video_batch_pipeline.py

**在 initialize_batch_system() 中**：
- 添加 `_create_modules()` 调用确保模块已创建
- 收集 `saver_input_queues` (pipeline.queue3)
- 将 Reader 输出队列注册到 `MultiVideoQueue.queues[idx]`
- 传递 `output_queues` 给 `TrackerDispatcher`

```python
# 收集Saver输入队列
saver_input_queues.append(pipeline.queue3)

# 将Reader输出队列注册到MultiVideoQueue
self.multi_video_queue.queues[idx] = reader_output_queue

# 创建TrackerDispatcher时传入output_queues
self.tracker_dispatcher = TrackerDispatcher(
    input_queues=batch_output_queues,
    tracker_instances=tracker_instances,
    output_queues=saver_input_queues,  # 追踪结果发送到Saver
    num_videos=num_videos,
    stop_event=self.stop_event
)
```

**创建 Pipeline 时跳过追踪线程**：
```python
pipeline = SingleVideoPipeline(
    ...
    skip_inference=True,  # 跳过单独的推理线程
    skip_tracker=True     # 跳过单独的追踪线程，由TrackerDispatcher处理
)
```

### 3. 修复 single_video_pipeline.py

**添加 skip_tracker 参数**：
```python
def __init__(self, ..., skip_tracker: bool = False):
    self.skip_tracker = skip_tracker
```

**start() 方法支持跳过 Tracker 线程**：
```python
if not self.skip_tracker and self.tracker:
    self.tracker_thread = threading.Thread(
        target=self.tracker.run,
        name=f"{self.pipeline_id}-Tracker",
        daemon=False
    )
```

### 4. 修复 main.py

**每个视频创建独立的追踪器**：
```python
for i, source in enumerate(video_sources):
    # 每个视频创建独立的追踪器实例
    video_tracker = ByteTracker(
        model_path=config.model_path,
        device=config.device,
        track_high_thresh=config.track_high_thresh,
        track_low_thresh=config.track_low_thresh,
        track_buffer=config.track_buffer,
        frame_rate=30.0
    )
    
    system.create_pipeline(
        video_source=source,
        tracker_instance=video_tracker,
        save_func=save_func,
        pipeline_id=pipeline_id
    )
```

## 修复后的数据流

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MultiVideoWithBatchPipeline                       │
│                                                                      │
│  Video0:                                                             │
│  ┌────────┐    ┌──────────────────┐                                 │
│  │ Reader │───▶│ reader_queue[0]  │───┐                             │
│  └────────┘    └──────────────────┘   │                             │
│                                        │                             │
│  Video1:                               │                             │
│  ┌────────┐    ┌──────────────────┐   │                             │
│  │ Reader │───▶│ reader_queue[1]  │───┼───▶ MultiVideoQueue         │
│  └────────┘    └──────────────────┘   │           │                 │
│                                        │           │                 │
│  ...                                   │           ▼                 │
│                                                                      │
│                                      ┌─────────────────┐             │
│                                      │ BatchInferencer │             │
│                                      │   (批推理)      │             │
│                                      └────────┬────────┘             │
│                                               │                      │
│                     ┌─────────────────────────┼─────────────┐        │
│                     │                         │             │        │
│                     ▼                         ▼             ▼        │
│          ┌──────────────────┐      ┌──────────────────┐  ...        │
│          │ batch_queue[0]   │      │ batch_queue[1]   │             │
│          └────────┬─────────┘      └────────┬─────────┘             │
│                   │                         │                        │
│                   └────────────┬────────────┘                        │
│                                │                                     │
│                                ▼                                     │
│                     ┌─────────────────────┐                          │
│                     │ TrackerDispatcher   │                          │
│                     │   (按序追踪1...n)   │                          │
│                     └─────────┬───────────┘                          │
│                               │                                      │
│             ┌─────────────────┼─────────────────┐                    │
│             │                 │                 │                    │
│             ▼                 ▼                 ▼                    │
│  ┌──────────────────┐ ┌──────────────────┐                          │
│  │ saver_queue[0]   │ │ saver_queue[1]   │  ...                     │
│  │ (pipeline.queue3)│ │ (pipeline.queue3)│                          │
│  └────────┬─────────┘ └────────┬─────────┘                          │
│           │                    │                                     │
│           ▼                    ▼                                     │
│      ┌─────────┐          ┌─────────┐                                │
│      │ Saver 0 │          │ Saver 1 │  ...                          │
│      └────┬────┘          └────┬────┘                                │
│           │                    │                                     │
│           ▼                    ▼                                     │
│    video_0_tracked.mp4  video_1_tracked.mp4                          │
└─────────────────────────────────────────────────────────────────────┘
```

## 关键线程

1. **Reader线程** × n: 每个视频一个，读取帧到 reader_queue
2. **BatchInferencer线程** × 1: 批推理，从 MultiVideoQueue 取帧
3. **TrackerDispatcher线程** × 1: 按序追踪（内部有 buffer_fill 和 tracking_dispatch 子线程）
4. **Saver线程** × n: 每个视频一个，保存结果

## 相关文件
- src/batch_inference_system.py
- src/multi_video_batch_pipeline.py
- src/single_video_pipeline.py
- src/main.py

## 验证命令
```bash
python main.py -i ../videos/video00.mp4 ../videos/video01.mp4 -m ../model/yolo12n.engine -d cuda --batch-size 32 --no-frames
```

预期结果：
- 生成 `result/video_0/video_0_tracked.mp4`
- 生成 `result/video_1/video_1_tracked.mp4`
