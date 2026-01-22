import cv2
from collections import defaultdict
import queue
import os
from concurrent.futures import ThreadPoolExecutor
import threading
class Video_Handler:
    def __init__(self, capacity, path_list, stop_event):
        # 增加队列大小以防止死锁（原来的1000太小）
        # 计算理论最大队列需求：预留足够空间防止读取线程和消费者线程互相阻塞
        queue_size = max(5000, capacity * 2)
        self.__buffer = queue.Queue(maxsize=queue_size)
        self.capacity = capacity
        self.path_list = path_list
        self.stop_event = stop_event
        self.pool = ThreadPoolExecutor(max_workers=len(path_list))
        # 为每个视频源维护一个帧计数器
        self.frame_counters = {path: 0 for path in path_list}
        # 记录每个视频的总帧数（用于动态线程管理）
        self.video_lengths = {}  # {path: total_frames}
        self.finished_videos = set()  # 已完成的视频集合
        self.active_threads = set(path_list)  # 活跃线程集合
        self.lock = threading.Lock()

    def __clear_buffer(self):
        while not self.__buffer.empty():
            try:
                self.__buffer.get_nowait()
                self.__buffer.task_done()
            except queue.Empty:
                pass

    def __worker_loop(self, path):
        cap = cv2.VideoCapture(path)
        try:
            if not cap.isOpened():
                print(f"Failed to open {path}")
                return
            
            # 获取视频总帧数
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            with self.lock:
                self.video_lengths[path] = total_frames
            print(f"[Reader] {os.path.basename(path)}: {total_frames} 帧")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    # 视频读完了，标记为已完成
                    with self.lock:
                        self.finished_videos.add(path)
                        self.active_threads.discard(path)
                        finished_count = len(self.finished_videos)
                        active_count = len(self.active_threads)
                    
                    print(f"[Reader] {os.path.basename(path)} 读取完毕 "
                          f"(已完成: {finished_count}/{len(self.path_list)}, 活跃: {active_count})")
                    break 
                
                # 为帧分配编号（1-based）
                self.frame_counters[path] += 1
                frame_id = self.frame_counters[path]
                
                try:
                    # 将帧数据、路径和帧编号一起放入队列
                    self.__buffer.put((frame, path, frame_id), timeout=0.5)
                except queue.Full:
                    continue # 队列满则重试
                
                # 检查停止信号，但继续处理已读取的帧
                if self.stop_event.is_set():
                    break
        finally:
            cap.release()
            print(f"[Reader] {os.path.basename(path)} 线程关闭")

    def read_video(self):
        try:
            for path in self.path_list: 
                if not os.path.exists(path):raise FileNotFoundError(f"{path} does not exist!")
            _ = [self.pool.submit(self.__worker_loop, path) for path in self.path_list]
        except Exception as e:
            print(f"Manager error: {e}")
    
    def get_reading_status(self):
        """获取视频读取状态"""
        with self.lock:
            return {
                'finished_videos': list(self.finished_videos),
                'active_threads': list(self.active_threads),
                'frame_counts': dict(self.frame_counters)
            }
    
    def getbuffer(self):
        return self.__buffer

# if __name__ == "__main__":
#     paths = ["../videos/video00.mp4", "../videos/video01.mp4"]
#     handler = Video_Handler(5000, path_list=paths) # 减小容量以防内存溢出
    
#     # 启动读取
#     handler.read_video()
#     buffer = handler.getbuffer()

#     if not os.path.exists("../result"):
#         os.makedirs("../result", exist_ok=True)

#     print(f"Total frames captured: {len(buffer)}")

#     src_frameCount = defaultdict(int)
#     for frame,path in buffer:
#         file_name = os.path.basename(path)
#         src_frameCount[file_name] += 1
#         count = src_frameCount[file_name]
#         cv2.imwrite(f"../result/img_{count}_{file_name}.jpg", frame)