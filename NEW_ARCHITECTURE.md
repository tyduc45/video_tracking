# 新架构设计：单一视频单一流水线模式

## 核心理念

摒弃多视频源同时读入+批处理的方案，采用**单一视频单一流水线**的架构：
- 每个视频对应一条独立的Pipeline
- 每条Pipeline内部使用**生产者-消费者模型**完全解耦各个阶段
- 多条Pipeline并行执行，互不干扰（类似Docker容器隔离）

## 架构总体设计

```
┌─────────────────────────────────────────────────────────────┐
│                      PipelineManager                        │
│  负责创建、管理、协调多条Pipeline的生命周期                  │
└─────────────────────────────────────────────────────────────┘
         ↓
    ┌────────────────────────────────────┐
    │  Pipeline0 (Video0)                │
    │  ┌──────────────────────────────┐  │
    │  │ Producer  Consumer Consumer │  │
    │  │ (Reader)  (Infer) (Track)   │  │
    │  │    ↓         ↓        ↓     │  │
    │  │  Queue1   Queue2   Queue3  │  │
    │  │    ↓         ↓        ↓     │  │
    │  │  [Output Saver]             │  │
    │  └──────────────────────────────┘  │
    └────────────────────────────────────┘
         ↓
    ┌────────────────────────────────────┐
    │  Pipeline1 (Video1)                │
    │  ┌──────────────────────────────┐  │
    │  │ Producer  Consumer Consumer │  │
    │  │ (Reader)  (Infer) (Track)   │  │
    │  │    ↓         ↓        ↓     │  │
    │  │  Queue1   Queue2   Queue3  │  │
    │  │    ↓         ↓        ↓     │  │
    │  │  [Output Saver]             │  │
    │  └──────────────────────────────┘  │
    └────────────────────────────────────┘
         ↓
         ...
```

## 核心模块设计

### 1. VideoSource - 视频源抽象

```python
from abc import ABC, abstractmethod
from typing import Optional, Tuple

class VideoSource(ABC):
    """视频源抽象基类"""
    
    @abstractmethod
    def open(self) -> bool:
        """打开视频源"""
        pass
    
    @abstractmethod
    def read(self) -> Tuple[bool, np.ndarray]:
        """读取一帧，返回 (success, frame)"""
        pass
    
    @abstractmethod
    def get_fps(self) -> float:
        """获取视频FPS"""
        pass
    
    @abstractmethod
    def get_frame_count(self) -> int:
        """获取总帧数（若未知则返回-1）"""
        pass
    
    @abstractmethod
    def get_resolution(self) -> Tuple[int, int]:
        """获取分辨率 (width, height)"""
        pass
    
    @abstractmethod
    def close(self):
        """关闭视频源"""
        pass

# 实现类
class LocalVideoSource(VideoSource):
    """本地视频文件"""
    pass

class WebcamSource(VideoSource):
    """网络摄像头/Webcam"""
    pass
```

### 2. SingleVideoPipeline - 单一视频流水线

```python
class SingleVideoPipeline:
    """
    单一视频的完整处理流水线
    
    架构：
    VideoSource → [Reader] ↓
                  Queue1 ↓
                  [Inferencer] ↓
                  Queue2 ↓
                  [Tracker] ↓
                  Queue3 ↓
                  [OutputSaver]
    """
    
    def __init__(self, video_source: VideoSource, pipeline_id: str):
        self.video_source = video_source
        self.pipeline_id = pipeline_id
        
        # 三个队列
        self.queue1 = Queue()  # Reader → Inferencer
        self.queue2 = Queue()  # Inferencer → Tracker
        self.queue3 = Queue()  # Tracker → OutputSaver
        
        # 四个处理线程
        self.reader_thread = None
        self.inferencer_thread = None
        self.tracker_thread = None
        self.saver_thread = None
        
        # 控制信号
        self.stop_event = threading.Event()
        self.statistics = {}
    
    def start(self):
        """启动流水线，创建所有工作线程"""
        pass
    
    def stop(self):
        """停止流水线，等待所有线程完成"""
        pass
    
    def get_statistics(self):
        """获取流水线的统计信息"""
        pass
```

### 3. PipelineManager - 管理器

```python
class PipelineManager:
    """
    管理多条独立的Pipeline
    """
    
    def __init__(self):
        self.pipelines = {}  # pipeline_id → SingleVideoPipeline
        self.lock = threading.Lock()
    
    def create_pipeline(self, video_source: VideoSource) -> str:
        """创建一条新的Pipeline，返回pipeline_id"""
        pass
    
    def start_all(self):
        """启动所有Pipeline"""
        pass
    
    def stop_all(self):
        """停止所有Pipeline"""
        pass
    
    def wait_all(self):
        """等待所有Pipeline完成"""
        pass
    
    def get_status(self):
        """获取所有Pipeline的状态"""
        pass
```

### 4. Producer-Consumer 框架

每条Pipeline包含3个Queue和4个处理阶段：

```
┌─────────────┐
│   Reader    │  生产者：读取视频帧
│  (Producer) │
└──────┬──────┘
       │ (Raw Frames)
    Queue1
       │
   ┌───▼──────┐
   │Inferencer│  消费者+生产者：YOLO推理
   │(Consumer)│
   │(Producer)│
   └───┬──────┘
       │ (Detections)
    Queue2
       │
   ┌───▼───────┐
   │ Tracker   │  消费者+生产者：ByteTrack追踪
   │(Consumer) │
   │(Producer) │
   └───┬───────┘
       │ (Tracked Results)
    Queue3
       │
   ┌───▼──────┐
   │  Saver   │  消费者：保存结果
   │(Consumer)│
   └──────────┘
```

## 工作流程

### 示例：处理2个视频源

```
输入：
  - /path/to/video1.mp4
  - rtsp://camera.url/stream

1. PipelineManager 创建 2 条 Pipeline
   - Pipeline0: video1.mp4
   - Pipeline1: rtsp://camera.url/stream

2. 各Pipeline并行运行
   时刻T0:
     Pipeline0: 读取frame_1 → 推理frame_1 → 追踪frame_1
     Pipeline1: 读取frame_1 → 推理frame_1 → 追踪frame_1
   
   时刻T1:
     Pipeline0: 读取frame_2 → 推理frame_2 → 追踪frame_2 → 保存frame_1
     Pipeline1: 读取frame_2 → 推理frame_2 → 追踪frame_2 → 保存frame_1

3. 各自独立输出
   result/video1/
     ├── frame_001.jpg
     ├── frame_002.jpg
     └── video1_tracked.mp4
   
   result/stream/
     ├── frame_001.jpg
     ├── frame_002.jpg
     └── stream_tracked.mp4
```

## 视频源的抽象

### 支持的源类型

```python
# 1. 本地文件
source1 = LocalVideoSource("/path/to/video.mp4")

# 2. 网络直播
source2 = WebcamSource("rtsp://192.168.1.100/stream")
source3 = WebcamSource("http://192.168.1.100:8080/video")

# 3. 本地摄像头
source4 = WebcamSource(0)  # 0表示默认摄像头
```

### 源接口的优点

1. **统一接口**：所有源都实现相同接口
2. **易于扩展**：新增源类型只需继承VideoSource
3. **参数化**：可以在运行时切换源类型
4. **隔离变化**：Pipeline无需知道源的具体实现

## 优势对比

### 旧方案（多视频源同时读入+批处理）
- ❌ 高耦合：读取、推理、追踪、保存紧密耦合
- ❌ 复杂乱序恢复：多线程读取导致帧乱序，需要复杂同步机制
- ❌ 共享资源竞争：多视频共享推理队列，导致调度复杂
- ❌ 难以调试：问题难以定位到具体视频

### 新方案（单一视频单一流水线）
- ✅ **高内聚**：每条Pipeline独立完整
- ✅ **低耦合**：通过Queue解耦各阶段
- ✅ **天然有序**：单个视频的帧天然有序，无需乱序恢复
- ✅ **易于扩展**：添加新视频只需创建新Pipeline
- ✅ **容易监控**：每条Pipeline的性能独立可观察
- ✅ **容错性强**：单条Pipeline失败不影响其他
- ✅ **易于调试**：问题明确属于哪条Pipeline

## 配置和使用

### 基本使用

```python
from pipeline_manager import PipelineManager
from video_source import LocalVideoSource, WebcamSource

# 创建管理器
manager = PipelineManager()

# 添加视频源
manager.create_pipeline(LocalVideoSource("videos/video1.mp4"))
manager.create_pipeline(WebcamSource("rtsp://camera.url/stream"))

# 启动所有Pipeline
manager.start_all()

# 等待完成
manager.wait_all()

# 获取统计信息
print(manager.get_status())
```

## 目录结构规划

```
src/
├── video_source.py            # VideoSource 抽象类
├── local_video_source.py       # LocalVideoSource 实现
├── webcam_source.py            # WebcamSource 实现
│
├── pipeline_data.py            # 数据结构（FrameData等）
├── single_video_pipeline.py    # SingleVideoPipeline
├── pipeline_manager.py         # PipelineManager
│
├── modules/
│   ├── reader.py              # Reader（生产者）
│   ├── inferencer.py          # Inferencer（推理）
│   ├── tracker.py             # Tracker（追踪）
│   └── saver.py               # Saver（保存结果）
│
├── visualizer.py              # 可视化模块
├── config.py                  # 配置管理
│
└── main.py                    # 主程序入口
```

## 下一步

1. 实现 `VideoSource` 及其子类
2. 实现 `SingleVideoPipeline` 和队列系统
3. 实现各处理模块（Reader、Inferencer、Tracker、Saver）
4. 实现 `PipelineManager`
5. 编写主程序和示例
