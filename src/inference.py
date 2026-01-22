# push
import os
import time
import queue
import threading
from ultralytics import YOLO
import sys
from io import StringIO
from tracker_manager import TrackerManager, FrameData
from collections import defaultdict

class BatchInferencer(threading.Thread):
    def __init__(self, queue, batch_size, model_path, save_path, stop_event, video_paths):
        super().__init__(name="InferenceThread")
        self.daemon = True  # 设置为守护线程，主程序退出时自动杀死
        self.queue = queue
        self.batch_size = batch_size
        self.isRunning = True
        self.stop_event = stop_event
        self.save_path = save_path
        self.video_paths = video_paths
        self.model_path = model_path  # 保存模型路径供后续使用
        
        # 初始化追踪器管理器
        self.tracker_manager = TrackerManager(video_paths)
        
        # 1. 在加载前，先处理模型导出逻辑
        self.model = self._setup_model_with_history(model_path, task_type="detect")
        print(f"[Consumer] 批处理流水线就绪，当前 Batch Size: {self.batch_size}")

    def _setup_model_with_history(self, path, task_type):
        """
        基于历史记录文件管理 Engine 导出。
        逻辑：对比记录的 batch 与当前需求，不符则清理并重编。
        """
        base_path = os.path.splitext(path)[0]
        engine_path = f"{base_path}.engine"
        pt_path = f"{base_path}.pt"
        meta_path = f"{base_path}_batch.meta" # 记录 Batch Size 的小文件

        # 尝试读取历史 Batch 需求
        last_batch = None
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    last_batch = int(f.read().strip())
            except Exception:
                last_batch = None

        # 核心判断逻辑
        need_reexport = False
        if not os.path.exists(engine_path):
            print("[Setup] Engine 不存在，准备初次导出...")
            need_reexport = True
        elif last_batch != self.batch_size:
            print(f"[Setup] 需求变更：历史 Batch({last_batch}) -> 当前 Batch({self.batch_size})")
            need_reexport = True

        if need_reexport:
            # 清理历史残余
            self._cleanup_files(base_path)
            
            if not os.path.exists(pt_path):
                raise FileNotFoundError(f"找不到源码模型 {pt_path}，无法执行重编。")
            
            # 从 PT 导出新 Engine
            print(f"[Export] 正在根据新需求 ({self.batch_size}) 导出引擎...")
            raw_model = YOLO(pt_path)
            new_path = raw_model.export(
                format="engine",
                batch=self.batch_size,
                dynamic=True,
                half=True,
                device=0
            )
            
            # 更新历史记录文件
            with open(meta_path, 'w') as f:
                f.write(str(self.batch_size))
            
            return YOLO(new_path, task=task_type)
        
        else:
            print(f"[Success] 需求未变 (Batch={last_batch})，直接加载现有引擎。")
            return YOLO(engine_path, task=task_type)

    def _cleanup_files(self, base_path):
        """物理删除旧的导出文件"""
        for ext in ['.engine', '.onnx', '_batch.meta']:
            file_p = f"{base_path}{ext}"
            if os.path.exists(file_p):
                try:
                    os.remove(file_p)
                    print(f"[Cleanup] 已移除旧文件: {file_p}")
                except Exception as e:
                    print(f"[Cleanup] 无法删除 {file_p}: {e}")

    def run(self):
        """主循环：负责高层逻辑调度"""
        try:
            self._ensure_dir_exists()
            
            while self.isRunning:
                # 1. 收集数据
                batch_frames, batch_paths, batch_frame_ids, stop_signal = self._collect_batch()
                
                if not batch_frames:
                    if stop_signal: break
                    continue

                # 2. 执行推理
                actual_num = len(batch_frames)
                results = self._execute_inference(batch_frames)
                
                if results:
                    # 3. 构建 FrameData 对象并送入追踪管理器
                    tracked_results = self._process_with_tracking(
                        results, batch_paths, batch_frame_ids, batch_frames
                    )
                    
                    # 4. 保存结果
                    self._save_batch_results(tracked_results)
                
                if stop_signal or self.stop_event.is_set():
                    break
        finally:
            self._final_cleanup()

    def _ensure_dir_exists(self):
        """确保保存目录存在"""
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)
            print(f"[Consumer] 创建目录: {self.save_path}")

    def _collect_batch(self):
        """
        核心组装逻辑：从队列获取数据
        返回: (frames_list, paths_list, frame_ids_list, stop_signal_received)
        """
        frames, paths, frame_ids = [], [], []
        stop_signal = False
        
        for i in range(self.batch_size):
            try:
                # 第一帧给 1.0s 超时，后续帧给 0.1s 实现快速滑动
                timeout = 1.0 if i == 0 else 0.1
                data = self.queue.get(timeout=timeout)
                
                if data is None: # 收到结束标志
                    stop_signal = True
                    self.isRunning = False
                    break
                
                # 现在数据包含 (frame, path, frame_id)
                frames.append(data[0])
                paths.append(data[1])
                frame_ids.append(data[2])
                
            except queue.Empty:
                if self.stop_event.is_set():
                    stop_signal = True
                # 如果队列空且有数据，继续处理当前批次
                # 否则等待新数据
                if not frames:
                    continue
                break # 有数据时直接处理，不再等待
                
        return frames, paths, frame_ids, stop_signal

    def _execute_inference(self, frames):
        """执行推理，处理补齐逻辑"""
        try:
            # 补齐到固定 Batch Size 以适配 TensorRT Engine
            actual_num = len(frames)
            while len(frames) < self.batch_size:
                frames.append(frames[-1])

            return self.model.predict(
                source=frames, conf=0.25, device=0, verbose=False
            )
        except Exception as e:
            print(f"[Consumer] 推理崩溃: {e}")
            self.stop_event.set()
            return None

    def _process_with_tracking(self, results, batch_paths, batch_frame_ids, batch_frames):
        """
        处理推理结果，通过追踪管理器进行时序同步和追踪
        
        Args:
            results: YOLO 推理结果
            batch_paths: 对应的视频路径列表
            batch_frame_ids: 对应的帧编号列表
            batch_frames: 原始帧图像
            
        Returns:
            已排序并追踪的帧数据列表
        """
        tracked_results = []
        
        for idx, (result, path, frame_id, frame) in enumerate(zip(results, batch_paths, batch_frame_ids, batch_frames)):
            try:
                # 获取视频 ID
                video_id = self.tracker_manager.get_video_id(path)
                if not video_id:
                    print(f"[Tracking] 警告：无法识别视频路径 {path}")
                    continue
                
                # 创建 FrameData 对象
                frame_data = FrameData(
                    frame=frame,
                    path=path,
                    video_id=video_id,
                    frame_id=frame_id,
                    detections=result
                )
                
                # 通过追踪管理器处理（乱序恢复）
                ready_frames = self.tracker_manager.process_frame(frame_data)
                
                # 对已排序完成的帧进行追踪
                if ready_frames:
                    tracked_frames = self.tracker_manager.track_frames(
                        ready_frames, 
                        model_path=self.model_path  # 传递正确的模型路径
                    )
                    tracked_results.extend(tracked_frames)
                    
                    # 定期打印追踪状态
                    if len(tracked_results) % 10 == 0:
                        self.tracker_manager.print_status()
                        
            except Exception as e:
                print(f"[Tracking] 处理帧 {frame_id} 失败: {e}")
                import traceback
                traceback.print_exc()
        
        return tracked_results

    def _save_batch_results(self, results):
        """
        保存追踪结果到磁盘，同时维护内存缓冲区用于最后的视频生成和删除
        使用异步线程处理磁盘写入，防止阻塞消费者线程
        
        Args:
            results: FrameData 对象列表，包含追踪信息
        """
        # 初始化缓存字典（按视频ID分组）
        if not hasattr(self, 'frame_buffer'):
            self.frame_buffer = {}  # 用于记录帧数据，后续生成视频
        
        if not hasattr(self, 'frame_files'):
            self.frame_files = defaultdict(list)  # {video_id: [file_paths]}
        
        if not hasattr(self, 'disk_write_lock'):
            self.disk_write_lock = threading.Lock()  # 保护磁盘写入操作
        
        for frame_data in results:
            try:
                video_id = frame_data.video_id
                
                # 初始化该视频的缓冲区
                if video_id not in self.frame_buffer:
                    self.frame_buffer[video_id] = {}
                
                # 获取注释后的帧
                if frame_data.detections:
                    # 绘制检测框和追踪 ID
                    annotated_frame = frame_data.detections.plot()
                else:
                    annotated_frame = frame_data.frame
                
                # 先保存到内存（快速操作）
                self.frame_buffer[video_id][frame_data.frame_id] = annotated_frame
                
                # 异步保存到磁盘（使用后台线程，不阻塞主流程）
                write_thread = threading.Thread(
                    target=self._async_save_frame,
                    args=(annotated_frame, video_id, frame_data.frame_id, frame_data.path),
                    daemon=True
                )
                write_thread.start()
                
                # 定期输出统计信息
                total_frames = sum(len(frames) for frames in self.frame_buffer.values())
                if total_frames % 50 == 0:
                    print(f"[Consumer] 已缓存 {total_frames} 帧（异步保存到磁盘）")
                    
            except Exception as e:
                print(f"[Consumer] 缓存帧 {frame_data.frame_id} 失败: {e}")
    
    def _async_save_frame(self, frame, video_id, frame_id, frame_path):
        """
        异步保存单个帧到磁盘
        """
        try:
            with self.disk_write_lock:  # 防止并发写入冲突
                ts = int(time.time() * 1000)
                filename = os.path.basename(frame_path)
                save_name = os.path.join(
                    self.save_path, 
                    f"tracked_{ts}_{video_id}_f{frame_id}_{filename}.jpg"
                )
                
                import cv2
                cv2.imwrite(save_name, frame)
                
                # 记录文件路径（用于最后删除）
                if video_id in self.frame_files:
                    self.frame_files[video_id].append(save_name)
                else:
                    self.frame_files[video_id] = [save_name]
                
        except Exception as e:
            print(f"[Consumer] 异步保存帧 {frame_id} 失败: {e}")


    def _final_cleanup(self):
        """最后清理，从内存缓冲区生成视频"""
        print("[Consumer] 正在执行最终清理...")
        
        # 先清空队列
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        
        # 等待异步磁盘写入完成（给后台线程1秒时间）
        print("[Consumer] 等待异步磁盘写入完成...")
        import time as time_module
        time_module.sleep(1)
        
        # 刷新追踪器缓冲区，确保所有乱序帧都被输出
        remaining_frames = self.tracker_manager.flush_all_buffers(self.model_path)
        if remaining_frames:
            print(f"[Consumer] 从缓冲区刷新出 {len(remaining_frames)} 帧")
            self._save_batch_results(remaining_frames)
        
        # 生成视频
        if hasattr(self, 'frame_buffer') and self.frame_buffer:
            self._generate_videos_from_buffer()
        
        print("[Consumer] 推理线程安全退出")
    
    def _generate_videos_from_buffer(self):
        """
        从内存缓冲区生成视频文件，然后删除所有中间帧文件
        """
        import cv2
        
        # 创建快照以避免迭代过程中字典大小改变的错误
        # 必须在持有锁的情况下创建快照
        with self.disk_write_lock:
            video_ids_to_process = list(self.frame_buffer.keys())
        
        for video_id in video_ids_to_process:
            # 每次获取一个视频的帧数据副本（持有锁）
            with self.disk_write_lock:
                if video_id not in self.frame_buffer:
                    continue  # 该视频已被处理过
                frames_dict = self.frame_buffer[video_id].copy()
            
            if not frames_dict:
                continue
            
            try:
                # 按帧编号排序
                sorted_frame_ids = sorted(frames_dict.keys())
                
                # 获取第一帧以确定分辨率
                first_frame = frames_dict[sorted_frame_ids[0]]
                height, width = first_frame.shape[:2]
                
                # 创建视频写入器
                output_path = os.path.join(self.save_path, f"tracked_{video_id}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_path, fourcc, 30, (width, height))
                
                frame_count = 0
                for frame_id in sorted_frame_ids:
                    frame = frames_dict[frame_id]
                    writer.write(frame)
                    frame_count += 1
                
                writer.release()
                print(f"[VideoGen] 生成视频: {output_path} ({frame_count} 帧)")
                
                # 删除所有对应的中间帧文件
                if video_id in self.frame_files:
                    deleted_count = 0
                    for frame_file in self.frame_files[video_id]:
                        try:
                            if os.path.exists(frame_file):
                                os.remove(frame_file)
                                deleted_count += 1
                        except Exception as e:
                            print(f"[VideoGen] 删除文件失败 {frame_file}: {e}")
                    
                    print(f"[VideoGen] 已删除 {deleted_count} 个中间帧文件")
                
                # 清空内存中的帧数据（持有锁）
                with self.disk_write_lock:
                    if video_id in self.frame_buffer:
                        del self.frame_buffer[video_id]
                    if video_id in self.frame_files:
                        del self.frame_files[video_id]
                
            except Exception as e:
                print(f"[VideoGen] 生成 {video_id} 的视频失败: {e}")

    def stop(self):
        self.isRunning = False