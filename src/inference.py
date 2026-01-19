# push
import os
import time
import queue
import threading
from ultralytics import YOLO
import sys
from io import StringIO

class BatchInferencer(threading.Thread):
    def __init__(self, queue, batch_size, model_path, save_path, stop_event):
        super().__init__(name="InferenceThread")
        self.daemon = True  # 设置为守护线程，主程序退出时自动杀死
        self.queue = queue
        self.batch_size = batch_size
        self.isRunning = True
        self.stop_event = stop_event
        self.save_path = save_path
        
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
        try:
            # 确保保存目录存在
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path, exist_ok=True)
                print(f"[Consumer] 创建保存目录: {self.save_path}")
            
            while self.isRunning:
                batch_frames, batch_paths = [], []
                stop_signal_received = False
                first_wait = True

                # 2. 组装 Batch 逻辑 - 最多收集16帧，或在超时后进行推理
                for frame_idx in range(self.batch_size):
                    try:
                        # 第一帧需要等待（最多1秒），后续帧使用较短的超时（0.1秒）
                        timeout = 1.0 if first_wait else 0.1
                        data = self.queue.get(timeout=timeout)
                        first_wait = False
                        
                        if data is None: # 收到停止信号
                            stop_signal_received = True
                            self.isRunning = False
                            break
                        
                        frame, path = data
                        if frame is None: continue
                        
                        batch_frames.append(frame)
                        batch_paths.append(path)
                    except queue.Empty:
                        # 检查是否收到停止信号
                        if self.stop_event.is_set():
                            stop_signal_received = True
                            self.isRunning = False
                            break
                        
                        # 队列为空，如果已有帧则进行推理，否则继续等待
                        if batch_frames:
                            print(f"[Consumer] 队列超时，触发推理 ({len(batch_frames)} 帧)")
                            break
                        elif stop_signal_received:
                            break
                        else:
                            continue

                # 即使收到停止信号也处理已收集的帧
                if not batch_frames:
                    if stop_signal_received:
                        break
                    continue

                # 3. 动态适配推理
                actual_num = len(batch_frames)
                # 如果是静态 Engine，这里仍需 Padding，动态则不需要，为稳妥起见保留补齐
                while len(batch_frames) < self.batch_size:
                    batch_frames.append(batch_frames[-1])

                try:
                    # 再次检查是否应该停止
                    if self.stop_event.is_set():
                        break
                    
                    print(f"[Consumer] 处理批次: {actual_num} 帧 (补齐至 {len(batch_frames)})")
                    
                    # 捕获verbose输出
                    old_stdout = sys.stdout
                    sys.stdout = StringIO()
                    
                    try:
                        results = self.model.predict(
                            source=batch_frames, conf=0.25, device=0, verbose=True
                        )
                        verbose_output = sys.stdout.getvalue()
                    finally:
                        sys.stdout = old_stdout
                    
                    # 直接输出verbose结果
                    if verbose_output.strip():
                        print(verbose_output.strip())
                    
                    # 4. 适配 16 个图片的批处理保存
                    for i in range(actual_num):
                        orig_path = batch_paths[i]
                        # 仅提取文件名防止 Windows 路径错误
                        filename = os.path.basename(orig_path)
                        save_name = f"{self.save_path}/res_{int(time.time()*1000)}_{i}_{filename}.jpg"
                        try:
                            results[i].save(filename=save_name)
                        except Exception as save_err:
                            print(f"[Consumer] 保存失败 {save_name}: {save_err}")
                    
                    if stop_signal_received:
                        break
                        
                except Exception as e:
                    print(f"[Consumer] 推理错误: {e}")
                    self.stop_event.set()
                    # 不再 raise，直接退出循环
                    break
        finally:
            # 5. 最终清理内存，解救生产者
            print("[Consumer] 清理队列并退出推理线程...")
            while not self.queue.empty():
                try:
                    self.queue.get_nowait()
                except:
                    break
            print("[Consumer] 推理线程已退出")

    def stop(self):
        self.isRunning = False