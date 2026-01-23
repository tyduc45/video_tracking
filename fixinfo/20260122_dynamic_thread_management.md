# v1.0.3 动态线程管理与帧保存优化

**发布日期**: 2026-01-22  
**版本**: v1.0.3  
**状态**: ✅ 已实施

## 功能改进概览

两个关键功能已实现：

### ✅ 功能1：动态线程管理

**功能描述**: 当一个视频读取线程完成后，自动关闭该线程，允许长视频继续独占读取。

**实现细节**:

1. **视频长度检测**：在 `__worker_loop` 开始时，获取视频总帧数
   ```python
   total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
   ```

2. **完成标记**：视频读完时，标记为已完成并从活跃线程集合中移除
   ```python
   with self.lock:
       self.finished_videos.add(path)
       self.active_threads.discard(path)
   ```

3. **缓冲区特性**：
   - **前半段**: 两个视频的帧混合出现（两个线程并行读取）
   - **后半段**: 仅有长视频的帧（短视频线程已关闭）

**修改文件**:
- `src/video_reader.py`: 新增线程管理相关代码（+25行）

**修改清单**:
- 新增属性：`self.video_lengths` - 记录视频总帧数
- 新增属性：`self.finished_videos` - 已完成的视频集合
- 新增属性：`self.active_threads` - 活跃线程集合
- 新增属性：`self.lock` - 线程安全锁
- 修改方法：`__worker_loop()` - 添加完成检测逻辑
- 新增方法：`get_reading_status()` - 获取读取状态

**日志输出示例**:
```
[Reader] video0.mp4: 7200 帧
[Reader] video1.mp4: 4000 帧
[Reader] video1.mp4 读取完毕 (已完成: 1/2, 活跃: 1)
[Reader] video0.mp4 读取完毕 (已完成: 2/2, 活跃: 0)
[Reader] video0.mp4 线程关闭
[Reader] video1.mp4 线程关闭
```

---

### ✅ 功能2：帧保存和删除优化

**功能描述**: 允许先保存追踪图像到 result 目录，处理完毕后自动转换为视频并删除中间帧文件。

**实现流程**:

```
处理帧 → 保存到磁盘 (tracked_*.jpg)
         ↓
      保存到内存 (frame_buffer)
         ↓
程序结束 → 生成视频 (tracked_video_0.mp4)
         ↓
      删除中间帧文件
```

**修改文件**:
- `src/inference.py`: 改进帧保存和生成逻辑（+50行）

**修改清单**:

1. **导入 defaultdict**:
   ```python
   from collections import defaultdict
   ```

2. **修改 `_save_batch_results()` 方法**:
   - 同时保存到磁盘和内存
   - 记录文件路径用于后续删除
   ```python
   # 保存帧到磁盘
   cv2.imwrite(save_name, annotated_frame)
   
   # 在内存中记录帧数据
   self.frame_buffer[video_id][frame_data.frame_id] = annotated_frame
   
   # 记录文件路径
   self.frame_files[video_id].append(save_name)
   ```

3. **重新设计 `_generate_videos_from_buffer()` 方法**:
   - 从内存缓冲区生成视频
   - 生成后自动删除中间帧文件
   ```python
   # 生成视频
   writer.write(frame)
   
   # 删除帧文件
   os.remove(frame_file)
   
   # 清理内存数据
   del self.frame_buffer[video_id]
   del self.frame_files[video_id]
   ```

---

## 技术实现细节

### 实现1：线程管理机制

**类结构扩展**:
```python
class Video_Handler:
    def __init__(self, ...):
        # 原有属性
        self.frame_counters = {path: 0 for path in path_list}
        
        # 新增属性 (v1.0.3)
        self.video_lengths = {}           # 记录视频长度
        self.finished_videos = set()      # 已完成的视频
        self.active_threads = set(...)    # 活跃线程
        self.lock = threading.Lock()      # 线程安全
```

**完成检测逻辑**:
```python
def __worker_loop(self, path):
    # ... 读取视频头信息 ...
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    with self.lock:
        self.video_lengths[path] = total_frames
    
    while True:
        ret, frame = cap.read()
        if not ret:
            # 视频读完了
            with self.lock:
                self.finished_videos.add(path)
                self.active_threads.discard(path)
            break
        
        # ... 处理帧 ...
```

**状态查询接口**:
```python
def get_reading_status(self):
    """获取实时读取状态"""
    with self.lock:
        return {
            'finished_videos': list(self.finished_videos),
            'active_threads': list(self.active_threads),
            'frame_counts': dict(self.frame_counters)
        }
```

---

### 实现2：帧文件管理机制

**双层缓存设计**:
```python
class BatchInferencer:
    def _save_batch_results(self, results):
        # 初始化缓存
        if not hasattr(self, 'frame_buffer'):
            self.frame_buffer = {}  # 内存缓冲：{video_id: {frame_id: frame}}
        
        if not hasattr(self, 'frame_files'):
            self.frame_files = defaultdict(list)  # 文件路径：{video_id: [paths]}
        
        for frame_data in results:
            # 1. 保存到磁盘
            save_name = f"tracked_{video_id}_f{frame_id}.jpg"
            cv2.imwrite(save_name, annotated_frame)
            
            # 2. 保存到内存
            self.frame_buffer[video_id][frame_id] = annotated_frame
            
            # 3. 记录文件路径
            self.frame_files[video_id].append(save_name)
```

**清理机制**:
```python
def _generate_videos_from_buffer(self):
    for video_id, frames_dict in self.frame_buffer.items():
        # 1. 从内存生成视频
        writer = cv2.VideoWriter(output_path, ...)
        for frame_id in sorted_frame_ids:
            writer.write(frames_dict[frame_id])
        writer.release()
        
        # 2. 删除所有中间帧文件
        for frame_file in self.frame_files[video_id]:
            os.remove(frame_file)
        
        # 3. 清理内存数据
        del self.frame_buffer[video_id]
        del self.frame_files[video_id]
```

---

## 性能和存储特性

| 指标 | 说明 |
|------|------|
| **帧保存** | ✅ 支持保存中间追踪图像到 result 目录 |
| **检查点** | ✅ 可在处理过程中手动检查结果 |
| **自动清理** | ✅ 程序完毕后自动删除帧文件 |
| **最终输出** | ✅ 只保留视频文件，无冗余存储 |
| **线程管理** | ✅ 动态关闭完成的读取线程 |
| **缓冲特性** | ✅ 前半段混合，后半段单线程 |

---

## 使用示例

### 检查读取状态
```python
from src.video_reader import Video_Handler

handler = Video_Handler(capacity=1000, path_list=video_paths, stop_event=stop_event)
handler.read_video()

# 获取实时状态
status = handler.get_reading_status()
print(f"已完成: {status['finished_videos']}")
print(f"活跃线程: {status['active_threads']}")
print(f"帧计数: {status['frame_counts']}")
```

### 自动帧清理
```python
# 程序正常结束时，_final_cleanup() 会：
# 1. 刷新追踪器缓冲区
# 2. 保存所有帧
# 3. 生成视频文件
# 4. 删除中间帧文件

# 结果目录最终只包含：
# result/
#   ├── tracked_video_0.mp4 (7200帧视频)
#   ├── tracked_video_1.mp4 (4000帧视频)
#   └── (没有JPG文件)
```

---

## 日志示例

### 读取阶段
```
[Reader] video0.mp4: 7200 帧
[Reader] video1.mp4: 4000 帧
[Reader] 处理多视频帧...
[Reader] video1.mp4 读取完毕 (已完成: 1/2, 活跃: 1)
[Reader] 继续读取长视频...
[Reader] video0.mp4 读取完毕 (已完成: 2/2, 活跃: 0)
```

### 处理阶段
```
[Consumer] 并缓存 50 帧
[Consumer] 已保存并缓存 100 帧
[Consumer] 已保存并缓存 150 帧
...
```

### 生成阶段
```
[VideoGen] 生成视频: result/tracked_video_0.mp4 (7200 帧)
[VideoGen] 已删除 7200 个中间帧文件
[VideoGen] 生成视频: result/tracked_video_1.mp4 (4000 帧)
[VideoGen] 已删除 4000 个中间帧文件
```

---

## 文件变更清单

| 文件 | 修改内容 | 行数 |
|------|---------|------|
| `src/video_reader.py` | 新增线程管理相关代码 | +25 |
| `src/inference.py` | 修改帧保存和删除逻辑 | +50 |

---

## 兼容性

✅ **100% 向后兼容**
- 接口完全相同
- 用户代码无需修改
- 输出格式完全相同

---

## 版本演进

| 版本 | 日期 | 主要改进 |
|------|------|---------|
| v1.0.0 | 2026-01-22 | 初始完整版本 |
| v1.0.1 | 2026-01-22 | 修复 ByteTrack 初始化 |
| v1.0.2 | 2026-01-22 | 修复帧数统计和内存优化 |
| v1.0.3 | 2026-01-22 | **动态线程管理和帧保存优化** ← 当前 |

---

## 总结

v1.0.3 实现了两个关键功能：

1. **动态线程管理**: 短视频完成后自动关闭线程，长视频继续独占处理
   - 前半段: 两个视频并行读取
   - 后半段: 仅长视频独占线程

2. **帧保存和自动清理**: 
   - 支持保存中间追踪图像
   - 完毕后自动转换为视频
   - 自动删除所有中间帧文件
   - 最终只保留视频文件

🚀 **快速启动**: `python src/main.py`

