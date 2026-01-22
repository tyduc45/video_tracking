import time
import os
from video_reader import Video_Handler
from inference import BatchInferencer
import threading
import queue

def main():
    # 使用绝对路径而不是相对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    video_paths = [
        os.path.join(project_root, "videos", "video00.mp4"),
        os.path.join(project_root, "videos", "video01.mp4")
    ]
    model_path = os.path.join(project_root, "model", "yolo12n.pt")
    save_path = os.path.join(project_root, "result")
    
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
        print(f"[Main] 创建结果目录: {save_path}")

    stop_event = threading.Event()

    handler = Video_Handler(capacity=10000, path_list=video_paths,stop_event=stop_event)
    frame_queue = handler.getbuffer()

    # 3. 初始化推理器 (消费者)
    # 将同一个 queue 传给推理器，并传递视频路径用于追踪器初始化
    inferencer = BatchInferencer(
        queue=frame_queue, 
        batch_size=16, 
        model_path=model_path,
        save_path=save_path,
        stop_event=stop_event,
        video_paths=video_paths  # 传递视频路径列表给追踪管理器
    )

    print("--- 启动视频读取与推理流水线 ---")
    print(f"[Main] 视频路径: {video_paths}")
    print(f"[Main] 模型路径: {model_path}")
    print(f"[Main] 结果保存路径: {save_path}")
    
    # 4. 启动线程
    inferencer.start()    # 启动推理线程
    handler.read_video()  # 启动线程池读取视频
    

    try:
        # 等待视频读取完成（等待线程池）
        handler.pool.shutdown(wait=True)
        print("\n[Main] 所有视频读取完成")
        
        # 发送停止信号给推理器
        frame_queue.put(None)
        
        # 等待推理完成
        while inferencer.is_alive():
            q_size = frame_queue.qsize()
            print(f"当前队列堆积帧数: {q_size}", end='\r')
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\n[Main] 用户手动停止，正在下发停机指令...")
        stop_event.set()
        # 立即返回，由finally处理清理
        return
    finally:
        # 1. 第一时间按下全局红色按钮
        stop_event.set() 
        # 2. 确保队列中有停止信号
        try:
            frame_queue.put_nowait(None)
        except queue.Full:
            pass
        # 3. 告诉推理器停止
        inferencer.stop() 
        # 4. 设置 join 超时，防止僵死
        inferencer.join(timeout=2) 
        
        # 5. 如果推理线程仍然活着（被卡住），由于设置了daemon=True，主程序仍会退出
        if inferencer.is_alive():
            print("[Main] 推理线程无法正常退出（可能被推理卡住），主程序将强制退出")
        
        print("\n--- 流水线已彻底关闭 ---")

if __name__ == "__main__":
    main()