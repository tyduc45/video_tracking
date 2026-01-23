# 新架构快速开始指南

## 🚀 5分钟快速开始

### 1️⃣ 安装依赖

```bash
pip install opencv-python ultralytics numpy
```

### 2️⃣ 准备视频

```
videos/
  ├── video1.mp4
  └── video2.mp4
```

### 3️⃣ 运行处理

```bash
cd src
python main.py
```

✅ 自动生成: `result/video_0/`, `result/video_1/` 等

---

## 📚 核心概念

### Pipeline（流水线）

每个视频对应一条**独立的Pipeline**：

```
VideoSource (视频源) 
    ↓
[Reader] 读取帧
    ↓ Queue1
[Inferencer] YOLO推理
    ↓ Queue2  
[Tracker] ByteTrack追踪
    ↓ Queue3
[Saver] 保存结果
```

### VideoSource（视频源）- 高度抽象

支持多种视频源，使用统一接口：

```python
# 本地视频文件
source = LocalVideoSource("videos/video.mp4")

# 网络直播
source = WebcamSource("rtsp://camera.url/stream")
source = WebcamSource("http://ip:port/stream")

# 本地摄像头
source = WebcamSource(0)  # 0为默认摄像头
```

### 独立处理

每条Pipeline完全独立，互不干扰（如Docker容器）：

```
管理器
  ├─ Pipeline-0 (video0.mp4)
  │  ├─ Reader
  │  ├─ Inferencer  
  │  ├─ Tracker
  │  └─ Saver
  │
  ├─ Pipeline-1 (rtsp://camera/stream)
  │  ├─ Reader
  │  ├─ Inferencer
  │  ├─ Tracker
  │  └─ Saver
  │
  └─ Pipeline-2 (local_camera: 0)
     ├─ Reader
     ├─ Inferencer
     ├─ Tracker
     └─ Saver
```

---

## 💻 使用示例

### 基本使用

```python
from pipeline_manager import PipelineManager
from video_source import LocalVideoSource, WebcamSource

# 创建管理器
manager = PipelineManager(output_dir="result")

# 添加视频源
manager.create_pipeline(LocalVideoSource("videos/video1.mp4"))
manager.create_pipeline(WebcamSource("rtsp://192.168.1.100/stream"))

# 启动所有Pipeline
manager.start_all()

# 等待完成
manager.wait_all()

# 查看统计
manager.print_all_statistics()
```

### 自定义推理函数

```python
# 方式1：使用YOLO推理
from inference import YOLOInferencer

inferencer = YOLOInferencer(
    model_path="model/yolo12n.pt",
    device="cuda",
    confidence_threshold=0.5
)

manager.create_pipeline(
    video_source=source,
    inference_func=inferencer.infer,
    ...
)

# 方式2：自定义推理逻辑
def custom_inference(frame):
    """自定义推理函数"""
    # 实现自己的检测逻辑
    detections = [
        {
            'class_id': 0,
            'class_name': 'person',
            'confidence': 0.95,
            'bbox': [x1, y1, x2, y2],
        }
    ]
    return detections

manager.create_pipeline(
    video_source=source,
    inference_func=custom_inference,
    ...
)
```

### 自定义保存函数

```python
from visualizer import PipelineOutputHandler

# 创建输出处理器
output_handler = PipelineOutputHandler(
    output_dir="result",
    save_frames=True,
    save_video=True,
    draw_boxes=True,
    draw_ids=True,
)

def save_func(frame_data, output_dir):
    """自定义保存函数"""
    # 方式1：使用内置处理器
    output_handler.process_frame(frame_data)
    
    # 方式2：自定义处理
    # 在frame_data中访问：
    # - frame_data.frame        原始帧
    # - frame_data.detections   检测结果
    # - frame_data.tracks       追踪结果
    # - frame_data.frame_id     帧编号
    # - frame_data.video_id     视频ID

manager.create_pipeline(
    video_source=source,
    save_func=save_func,
    ...
)

# 生成最终视频
output_handler.generate_all_videos()
```

---

## 🎯 常见场景

### 场景1：处理多个本地视频

```python
from pathlib import Path

manager = PipelineManager(output_dir="result")

# 自动发现videos目录下的所有视频
for video_file in Path("videos").glob("*.mp4"):
    manager.create_pipeline(LocalVideoSource(str(video_file)))

manager.start_all()
manager.wait_all()
```

### 场景2：实时多摄像头监控

```python
manager = PipelineManager(output_dir="result", max_pipelines=4)

# 添加多个实时摄像头
cameras = [
    "rtsp://192.168.1.100/stream",
    "rtsp://192.168.1.101/stream",
    "rtsp://192.168.1.102/stream",
]

for url in cameras:
    manager.create_pipeline(WebcamSource(url))

manager.start_all()
manager.wait_all()  # 无限等待，适合持续监控
```

### 场景3：混合本地视频和网络流

```python
manager = PipelineManager()

# 本地视频
manager.create_pipeline(LocalVideoSource("videos/security.mp4"))

# 网络摄像头
manager.create_pipeline(WebcamSource("rtsp://entrance.cam/stream"))

# 本地摄像头
manager.create_pipeline(WebcamSource(0))

manager.start_all()
manager.wait_all()
```

### 场景4：性能优化（GPU推理）

```python
from config import Config
from inference import YOLOInferencer

config = Config(
    model_path="model/yolo12n.pt",
    device="cuda",              # 使用GPU
    confidence_threshold=0.6,   # 提高阈值减少计算
    batch_size=4,
)

inferencer = YOLOInferencer(
    model_path=config.model_path,
    device=config.device,
    confidence_threshold=config.confidence_threshold,
)

manager = PipelineManager(max_pipelines=10)

# 创建多个Pipeline共享同一个推理器
for i in range(5):
    manager.create_pipeline(
        video_source=LocalVideoSource(f"videos/video{i}.mp4"),
        inference_func=inferencer.infer,
    )

manager.start_all()
manager.wait_all()
```

---

## 📊 输出结构

```
result/
├── video_0/
│   ├── frames/
│   │   ├── frame_000001.jpg
│   │   ├── frame_000002.jpg
│   │   └── ...
│   └── video_0_tracked.mp4
│
├── video_1/
│   ├── frames/
│   │   └── ...
│   └── video_1_tracked.mp4
│
└── config.json
```

每个Pipeline的输出包含：
- `frames/`: 所有处理后的帧图像
- `{video_id}_tracked.mp4`: 最终输出视频
- 帧数据中包含检测框、追踪ID等信息

---

## 🔧 配置说明

### 完整配置示例

```python
from config import Config

config = Config(
    # 模型
    model_path="model/yolo12n.pt",
    model_type="yolov8",
    
    # 推理
    device="cuda",              # cuda 或 cpu
    confidence_threshold=0.5,
    iou_threshold=0.45,
    
    # Pipeline
    queue_size=10,              # 队列大小
    max_pipelines=10,           # 最多Pipeline数量
    
    # 追踪
    track_high_thresh=0.6,
    track_low_thresh=0.1,
    track_buffer=30,
    
    # I/O
    input_dir="videos",
    output_dir="result",
    
    # 可视化
    save_frames=True,
    save_video=True,
    save_fps=30.0,
    draw_boxes=True,
    draw_ids=True,
    draw_confidence=True,
    
    # 日志
    log_level="INFO",
    log_file="result/app.log",
)
```

---

## 🐛 常见问题

### Q: 为什么采用"单一视频单一流水线"架构？
**A**: 
- **独立性强**：每条Pipeline完全独立，互不干扰
- **易于扩展**：添加新视频只需创建新Pipeline
- **易于调试**：问题明确属于哪条Pipeline
- **性能可控**：每条Pipeline的性能可独立优化
- **容错性好**：单条Pipeline失败不影响其他

### Q: 支持哪些视频源？
**A**: 
- ✅ 本地视频文件（MP4、AVI、MKV等）
- ✅ RTSP网络直播流
- ✅ HTTP网络直播流
- ✅ 本地摄像头设备
- ✅ MJPEG直播流

### Q: 如何实现自定义推理？
**A**: 
```python
def my_inference(frame):
    # 实现自己的推理逻辑
    return detections

manager.create_pipeline(
    video_source=source,
    inference_func=my_inference,
)
```

### Q: 支持实时处理吗？
**A**: 完全支持，包括：
- 实时视频直播
- 实时摄像头输入
- 实时检测和追踪
- 流式输出结果

### Q: 如何实现并发处理多个视频？
**A**: Pipeline本身就是并发的，只需创建多个Pipeline：
```python
for source in video_sources:
    manager.create_pipeline(source)

manager.start_all()  # 所有Pipeline并行运行
```

---

## 📖 更多文档

- [NEW_ARCHITECTURE.md](NEW_ARCHITECTURE.md) - 详细架构设计
- [src/examples.py](src/examples.py) - 6个完整示例
- [src/config.py](src/config.py) - 配置管理

---

## 📞 技术支持

遇到问题？检查：
1. 日志文件：`result/app.log`
2. 配置验证：`config.validate()`
3. 源代码注释和文档字符串
