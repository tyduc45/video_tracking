import cv2
from collections import defaultdict
import queue
import os
from concurrent.futures import ThreadPoolExecutor
import threading
class Video_Handler:
    def __init__(self, capacity, path_list,stop_event):
        self.__buffer = queue.Queue(maxsize=1000)
        self.capacity = capacity
        self.path_list = path_list
        self.stop_event = stop_event
        self.pool = ThreadPoolExecutor(max_workers=len(path_list))

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
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break 
                try:
                    self.__buffer.put((frame, path), timeout=0.5)
                except queue.Full:
                    continue # 队列满则重试
                
                # 检查停止信号，但继续处理已读取的帧
                if self.stop_event.is_set():
                    break
        finally:
            cap.release()
            print(f"Video source released: {path}")

    def read_video(self):
        try:
            for path in self.path_list: 
                if not os.path.exists(path):raise FileNotFoundError(f"{path} does not exist!")
            _ = [self.pool.submit(self.__worker_loop, path) for path in self.path_list]
        except Exception as e:
            print(f"Manager error: {e}")
    
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