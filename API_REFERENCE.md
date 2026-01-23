# API文档 - 新架构

## 目录

1. [VideoSource 视频源](#videosource-视频源)
2. [SingleVideoPipeline 单一视频流水线](#singlevideoPipeline-单一视频流水线)
3. [PipelineManager 管理器](#pipelinemanager-管理器)
4. [推理和追踪](#推理和追踪)
5. [可视化输出](#可视化输出)
6. [数据结构](#数据结构)

---

## VideoSource 视频源

### 基类：VideoSource

视频源的抽象基类，所有具体实现都继承此类。

```python
class VideoSource(ABC):
    def __init__(self, source_id: str, name: str = None)
    def open(self) -> bool
    def read(self) -> Tuple[bool, Optional[np.ndarray]]
    def get_fps(self) -> float
    def get_frame_count(self) -> int
    def get_resolution(self) -> Tuple[int, int]
    def close(self)
    def is_open(self) -> bool
```

### LocalVideoSource 本地视频文件

```python
from video_source import LocalVideoSource

# 创建本地视频源
source = LocalVideoSource("videos/video.mp4")

# 打开和读取
source.open()
success, frame = source.read()
fps = source.get_fps()
total_frames = source.get_frame_count()
width, height = source.get_resolution()
source.close()

# 或使用Context Manager
with LocalVideoSource("videos/video.mp4") as source:
    success, frame = source.read()
```

**参数**:
- `file_path` (str): 本地视频文件路径

**返回值**:
- `read()`: 返回 `(success: bool, frame: np.ndarray or None)`

**异常**:
- 若文件不存在，`open()` 返回 False

---

### WebcamSource 网络摄像头

```python
from video_source import WebcamSource

# 本地摄像头
source = WebcamSource(0)  # 0为默认摄像头

# RTSP网络直播
source = WebcamSource("rtsp://192.168.1.100/stream")

# HTTP网络直播
source = WebcamSource("http://192.168.1.100:8080/video")

# 使用
source.open()
success, frame = source.read()
source.close()
```

**参数**:
- `source` (int | str):
  - int: 本地摄像头ID (0, 1, 2, ...)
  - str: URL地址 (rtsp://, http://, etc.)

**特点**:
- `get_frame_count()` 总返回 -1（直播流未知总帧数）
- `get_fps()` 根据摄像头或流的设置返回
- 自动设置缓冲区大小以减少延迟

---

## SingleVideoPipeline 单一视频流水线

完整的处理流水线，包含4个处理阶段和3个队列。

```python
from single_video_pipeline import SingleVideoPipeline
from video_source import LocalVideoSource

# 创建Pipeline
pipeline = SingleVideoPipeline(
    pipeline_id="video_0",
    video_source=LocalVideoSource("videos/video.mp4"),
    inference_func=my_inference_function,
    tracker_instance=my_tracker,
    save_func=my_save_function,
    output_dir="result",
    queue_size=10
)

# 启动Pipeline
pipeline.start()

# 等待完成
pipeline.wait(timeout=600)

# 获取统计信息
stats = pipeline.get_statistics()
pipeline.print_statistics()

# 停止Pipeline（若需要）
pipeline.stop()
```

### 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `pipeline_id` | str | Pipeline唯一标识 |
| `video_source` | VideoSource | 视频源对象 |
| `inference_func` | Callable | 推理函数 `(frame) -> detections` |
| `tracker_instance` | object | 追踪器实例（实现 `.update(detections)` 方法） |
| `save_func` | Callable | 保存函数 `(frame_data, output_dir) -> None` |
| `output_dir` | str | 输出目录 |
| `queue_size` | int | 各队列的大小限制 |

### 主要方法

#### `start()`
启动Pipeline，创建4个工作线程。

```python
pipeline.start()
```

#### `stop()`
停止Pipeline，发送停止信号。

```python
pipeline.stop()
```

#### `wait(timeout=None) -> bool`
等待Pipeline完成。

```python
# 无限等待
pipeline.wait()

# 超时等待
success = pipeline.wait(timeout=600)
if success:
    print("Pipeline completed")
```

#### `get_statistics() -> PipelineStatistics`
获取统计信息。

```python
stats = pipeline.get_statistics()
print(f"Total frames: {stats.total_frames}")
print(f"Dropped frames: {stats.dropped_frames}")
print(f"Inference FPS: {stats.average_inference_fps:.1f}")
print(f"Tracking FPS: {stats.average_tracking_fps:.1f}")
```

#### `print_statistics()`
打印详细的统计信息。

```python
pipeline.print_statistics()
```

---

## PipelineManager 管理器

管理多条独立的Pipeline，协调它们的生命周期。

```python
from pipeline_manager import PipelineManager
from video_source import LocalVideoSource

# 创建管理器
manager = PipelineManager(
    output_dir="result",
    max_pipelines=10
)

# 创建Pipeline
pipeline_id = manager.create_pipeline(
    video_source=LocalVideoSource("videos/video.mp4"),
    inference_func=my_inference,
    tracker_instance=my_tracker,
    save_func=my_save,
    pipeline_id="video_0"
)

# 启动所有Pipeline
manager.start_all()

# 等待所有完成
manager.wait_all(timeout=None)

# 查看状态
manager.print_status()
manager.print_all_statistics()
```

### 主要方法

#### `create_pipeline(...) -> str`
创建一条新Pipeline。

```python
pipeline_id = manager.create_pipeline(
    video_source=source,
    inference_func=infer_func,
    tracker_instance=tracker,
    save_func=save_func,
    pipeline_id="video_0"  # 可选，若无则自动生成
)
```

**返回**: Pipeline ID

#### `start_all()`
启动所有Pipeline。

```python
manager.start_all()
```

#### `stop_all()`
停止所有Pipeline。

```python
manager.stop_all()
```

#### `wait_all(timeout=None) -> bool`
等待所有Pipeline完成。

```python
success = manager.wait_all(timeout=600)
```

#### `get_pipeline(pipeline_id) -> SingleVideoPipeline`
获取指定的Pipeline。

```python
pipeline = manager.get_pipeline("video_0")
```

#### `get_all_pipelines() -> List[Tuple[str, SingleVideoPipeline]]`
获取所有Pipeline。

```python
all_pipelines = manager.get_all_pipelines()
for pid, pipeline in all_pipelines:
    print(f"{pid}: {pipeline}")
```

#### `get_status() -> Dict`
获取所有Pipeline的状态（结构化数据）。

```python
status = manager.get_status()
print(status)
# {
#   'timestamp': '2026-01-23T...',
#   'total_pipelines': 2,
#   'pipelines': {
#     'video_0': {
#       'state': 'running',
#       'total_frames': 100,
#       'dropped_frames': 0,
#       ...
#     }
#   }
# }
```

#### `print_status()`
打印所有Pipeline的状态（可读格式）。

```python
manager.print_status()
```

#### `print_all_statistics()`
打印所有Pipeline的详细统计信息。

```python
manager.print_all_statistics()
```

#### `reset()`
清空所有Pipeline。

```python
manager.reset()
```

---

## 推理和追踪

### YOLOInferencer YOLO推理器

```python
from inference import YOLOInferencer
import numpy as np

# 创建推理器
inferencer = YOLOInferencer(
    model_path="model/yolo12n.pt",
    device="cuda",  # 或 "cpu"
    confidence_threshold=0.5,
    iou_threshold=0.45
)

# 执行推理
frame = np.ndarray(...)  # BGR格式
detections = inferencer.infer(frame)

# detections 格式：
# [
#   {
#     'class_id': 0,
#     'class_name': 'person',
#     'confidence': 0.95,
#     'bbox': [x1, y1, x2, y2],  # 左上右下坐标
#   },
#   ...
# ]
```

### ByteTracker 追踪器

```python
from inference import ByteTracker

# 创建追踪器
tracker = ByteTracker(
    track_high_thresh=0.6,
    track_low_thresh=0.1,
    track_buffer=30,
    frame_rate=30.0
)

# 更新追踪
detections = [...]  # 推理结果
tracks = tracker.update(detections)

# tracks 格式：
# [
#   {
#     'track_id': 1,
#     'class_id': 0,
#     'class_name': 'person',
#     'confidence': 0.95,
#     'bbox': [x1, y1, x2, y2],
#     'state': 'tracked',  # 或 'lost'
#   },
#   ...
# ]
```

### ResultVisualizer 结果可视化

```python
from inference import ResultVisualizer
import numpy as np

# 绘制检测框
frame = np.ndarray(...)  # BGR
detections = [...]
result = ResultVisualizer.draw_detections(frame, detections)

# 绘制追踪框和ID
tracks = [...]
result = ResultVisualizer.draw_tracks(frame, tracks)

# 保存图像
ResultVisualizer.save_frame(result, "output/frame_001.jpg")
```

---

## 可视化输出

### PipelineOutputHandler 输出处理器

```python
from visualizer import PipelineOutputHandler

# 创建输出处理器
handler = PipelineOutputHandler(
    output_dir="result",
    save_frames=True,
    save_video=True,
    draw_boxes=True,
    draw_ids=True,
    draw_confidence=True,
    fps=30.0
)

# 处理每一帧
# (通常在Pipeline的save_func中调用)
def save_func(frame_data, output_dir):
    handler.process_frame(frame_data)

# 生成最终视频
handler.generate_all_videos()
```

### VideoGenerator 视频生成器

```python
from visualizer import VideoGenerator

# 创建视频生成器
generator = VideoGenerator(fps=30.0, codec='mp4v')

# 从帧目录生成视频
success = generator.generate_video(
    frame_dir="result/video_0/frames",
    video_path="result/video_0/video_0_tracked.mp4",
    resolution=(1920, 1080)
)
```

---

## 数据结构

### FrameData 帧数据

流经Pipeline的数据结构。

```python
@dataclass
class FrameData:
    frame_id: int                       # 帧序号 (1-based)
    timestamp: float                    # 时间戳
    frame: np.ndarray                   # 原始帧图像 (BGR格式)
    video_id: str                       # 视频源ID
    video_name: str                     # 视频源名称
    
    # 推理结果
    detections: Optional[List[Any]] = None
    
    # 追踪结果
    tracks: Optional[List[Any]] = None
    tracked_frame: Optional[np.ndarray] = None
```

**访问示例**:
```python
def my_save_func(frame_data, output_dir):
    frame_id = frame_data.frame_id
    frame = frame_data.frame
    detections = frame_data.detections
    tracks = frame_data.tracks
```

### PipelineStatistics 统计信息

Pipeline的统计数据。

```python
@dataclass
class PipelineStatistics:
    pipeline_id: str
    total_frames: int = 0
    dropped_frames: int = 0
    inference_time: float = 0.0
    tracking_time: float = 0.0
    average_inference_fps: float = 0.0
    average_tracking_fps: float = 0.0
    processing_state: str = "idle"  # idle, running, completed, failed, timeout
    error_message: Optional[str] = None
```

---

## 配置管理

### Config 配置类

```python
from config import Config, load_config, save_config

# 创建配置
config = Config(
    model_path="model/yolo12n.pt",
    device="cuda",
    confidence_threshold=0.5,
    output_dir="result",
    ...
)

# 验证配置
if config.validate():
    print("Configuration is valid")

# 转换为字典
config_dict = config.to_dict()

# 加载配置文件
config = load_config("config.json")

# 保存配置文件
save_config(config, "config.json")
```

**主要配置项**:
- `model_path`: YOLO模型路径
- `device`: cuda或cpu
- `confidence_threshold`: 置信度阈值 [0, 1]
- `queue_size`: Pipeline队列大小
- `max_pipelines`: 最多Pipeline数量
- `save_frames`: 是否保存帧
- `save_video`: 是否生成视频
- `output_dir`: 输出目录

---

## 完整使用流程

```python
from pipeline_manager import PipelineManager
from video_source import LocalVideoSource, WebcamSource
from inference import YOLOInferencer, ByteTracker
from visualizer import create_output_handler
from config import load_config

# 1. 加载配置
config = load_config()

# 2. 创建推理和追踪
inferencer = YOLOInferencer(
    model_path=config.model_path,
    device=config.device
)
tracker = ByteTracker()

# 3. 创建输出处理器
output_handler = create_output_handler(config)
def save_func(frame_data, output_dir):
    output_handler.process_frame(frame_data)

# 4. 创建管理器
manager = PipelineManager(output_dir=config.output_dir)

# 5. 添加视频源
sources = [
    LocalVideoSource("videos/video1.mp4"),
    WebcamSource("rtsp://camera.url/stream"),
]

for i, source in enumerate(sources):
    manager.create_pipeline(
        video_source=source,
        inference_func=inferencer.infer,
        tracker_instance=tracker,
        save_func=save_func,
        pipeline_id=f"video_{i}"
    )

# 6. 运行
manager.start_all()
manager.wait_all()

# 7. 输出
output_handler.generate_all_videos()
manager.print_all_statistics()
```

---

## 常见模式

### 模式1：自定义推理函数

```python
def custom_inference(frame):
    # 自定义推理逻辑
    # 返回检测结果列表
    return detections

manager.create_pipeline(
    video_source=source,
    inference_func=custom_inference,
)
```

### 模式2：自定义保存函数

```python
def custom_save(frame_data, output_dir):
    # 自定义保存逻辑
    # 访问frame_data的各个字段
    frame = frame_data.frame
    detections = frame_data.detections
    tracks = frame_data.tracks

manager.create_pipeline(
    save_func=custom_save,
)
```

### 模式3：并行处理多个视频

```python
for source in video_sources:
    manager.create_pipeline(video_source=source, ...)

manager.start_all()
manager.wait_all()
```

### 模式4：动态添加Pipeline

```python
# 运行时动态添加
new_pipeline_id = manager.create_pipeline(source)

# 获取并单独控制
pipeline = manager.get_pipeline(new_pipeline_id)
pipeline.start()
pipeline.wait()
```
