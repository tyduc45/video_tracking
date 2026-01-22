"""
完整的使用示例和最佳实践指南
"""
import os
import sys
import time
from pathlib import Path

# 导入模块
from video_reader import Video_Handler
from inference import BatchInferencer
from video_visualizer import visualize_results
import threading
import queue


def example_basic_usage():
    """
    基本使用示例：完整的多视频追踪流水线
    """
    print("\n" + "="*60)
    print("示例 1: 基本使用 - 多视频追踪")
    print("="*60)
    
    # 获取项目路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # 配置视频路径
    video_paths = [
        os.path.join(project_root, "videos", "video0.mp4"),
        os.path.join(project_root, "videos", "video1.mp4")
    ]
    
    # 配置模型和输出路径
    model_path = os.path.join(project_root, "model", "yolo12n.pt")
    result_dir = os.path.join(project_root, "result")
    
    # 检查文件是否存在
    for path in video_paths:
        if not os.path.exists(path):
            print(f"⚠️  警告：视频文件不存在 {path}")
    
    if not os.path.exists(model_path):
        print(f"⚠️  警告：模型文件不存在 {model_path}")
        return
    
    os.makedirs(result_dir, exist_ok=True)
    
    # 创建事件
    stop_event = threading.Event()
    
    # 1. 初始化视频读取器
    print("\n[Step 1] 初始化视频读取器...")
    handler = Video_Handler(capacity=1000, path_list=video_paths, stop_event=stop_event)
    frame_queue = handler.getbuffer()
    
    # 2. 初始化推理器（包含追踪）
    print("[Step 2] 初始化推理器和追踪系统...")
    inferencer = BatchInferencer(
        queue=frame_queue,
        batch_size=16,
        model_path=model_path,
        save_path=result_dir,
        stop_event=stop_event,
        video_paths=video_paths
    )
    
    print("\n[Step 3] 启动处理流水线...")
    print(f"  - 视频数量: {len(video_paths)}")
    print(f"  - 推理批大小: 16")
    print(f"  - 结果保存目录: {result_dir}")
    
    # 启动线程
    inferencer.start()
    handler.read_video()
    
    try:
        # 等待视频读取完成
        handler.pool.shutdown(wait=True)
        print("\n✓ 所有视频读取完成")
        
        # 发送停止信号给推理器
        frame_queue.put(None)
        
        # 等待推理完成
        processed_frames = 0
        while inferencer.is_alive():
            q_size = frame_queue.qsize()
            print(f"  处理中... 队列堆积: {q_size} 帧", end='\r')
            time.sleep(0.5)
        
        print("\n✓ 推理和追踪完成")
        
    except KeyboardInterrupt:
        print("\n⚠️  用户中断")
        stop_event.set()
        return
    finally:
        stop_event.set()
        try:
            frame_queue.put_nowait(None)
        except queue.Full:
            pass
        inferencer.stop()
        inferencer.join(timeout=2)
    
    # 3. 生成最终视频
    print("\n[Step 4] 生成追踪视频...")
    try:
        videos = visualize_results(
            frame_dir=result_dir,
            output_dir=os.path.join(result_dir, "videos"),
            fps=30
        )
        print(f"✓ 生成了 {len(videos)} 个视频文件")
    except Exception as e:
        print(f"⚠️  视频生成失败: {e}")
    
    print("\n" + "="*60)
    print("处理完成！")
    print("="*60)


def example_single_video_tracking():
    """
    示例 2: 单个视频追踪（用于测试）
    """
    print("\n" + "="*60)
    print("示例 2: 单个视频追踪测试")
    print("="*60)
    
    from tracker_manager import TrackerManager, FrameData
    import cv2
    import numpy as np
    
    # 创建模拟帧
    def create_dummy_frame(frame_id):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, f"Frame {frame_id}", (50, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame
    
    # 初始化追踪管理器
    video_paths = ["video0.mp4"]
    tracker_manager = TrackerManager(video_paths)
    
    print("模拟向追踪器发送乱序帧...")
    
    # 模拟乱序的帧序列
    frame_sequence = [1, 3, 2, 5, 4, 6, 7, 8]
    
    for frame_id in frame_sequence:
        frame = create_dummy_frame(frame_id)
        frame_data = FrameData(
            frame=frame,
            path="video0.mp4",
            video_id="video_0",
            frame_id=frame_id,
            detections=None
        )
        
        ready_frames = tracker_manager.process_frame(frame_data)
        
        print(f"  → 接收帧 {frame_id}: ", end="")
        if ready_frames:
            print(f"✓ 输出 {len(ready_frames)} 个已排序的帧 "
                  f"(ID: {[f.frame_id for f in ready_frames]})")
        else:
            print(f"⏳ 等待中（进入队列）")
        
        tracker_manager.print_status()
    
    print("\n✓ 乱序恢复演示完成")


def example_advanced_configuration():
    """
    示例 3: 高级配置 - 自定义参数
    """
    print("\n" + "="*60)
    print("示例 3: 高级配置")
    print("="*60)
    
    configs = {
        "batch_size": [8, 16, 32],
        "queue_capacity": [500, 1000, 2000],
        "fps": [15, 30, 60]
    }
    
    print("\n推荐配置组合：\n")
    
    print("1. 低延迟配置（实时处理）")
    print("   - batch_size=8")
    print("   - queue_capacity=500")
    print("   - 适用于实时监控场景")
    print()
    
    print("2. 均衡配置（通用）")
    print("   - batch_size=16")
    print("   - queue_capacity=1000")
    print("   - 适用于大多数应用")
    print()
    
    print("3. 高吞吐配置（离线处理）")
    print("   - batch_size=32")
    print("   - queue_capacity=2000")
    print("   - 适用于大规模离线处理")


def example_monitoring():
    """
    示例 4: 实时监控追踪状态
    """
    print("\n" + "="*60)
    print("示例 4: 追踪状态监控")
    print("="*60)
    
    from tracker_manager import TrackerManager
    
    video_paths = ["video0.mp4", "video1.mp4", "video2.mp4"]
    tracker_manager = TrackerManager(video_paths)
    
    print("\n追踪管理器初始化完成，各视频源状态：\n")
    
    status = tracker_manager.get_status()
    for video_id, info in status.items():
        print(f"  {video_id}:")
        print(f"    - 期望下一帧: {info['expected_frame_id']}")
        print(f"    - 乱序队列大小: {info['queue_size']}")
        print(f"    - 缓冲区大小: {info['buffer_size']}")
    
    print("\n🔍 监控指标说明：")
    print("  - 期望帧号: 追踪器期望的下一个帧编号")
    print("  - 乱序队列: 到达但时序不对的帧数")
    print("  - 缓冲区: 已排序完成可使用的帧数")


def print_menu():
    """打印菜单"""
    print("\n" + "="*60)
    print("YOLO追踪系统 - 使用示例菜单")
    print("="*60)
    print("1. 基本使用 - 完整的多视频追踪流水线")
    print("2. 单个视频追踪测试")
    print("3. 高级配置建议")
    print("4. 追踪状态监控演示")
    print("0. 退出")
    print("="*60)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        print_menu()
        choice = input("\n请选择 (0-4): ").strip()
    
    if choice == "1":
        example_basic_usage()
    elif choice == "2":
        example_single_video_tracking()
    elif choice == "3":
        example_advanced_configuration()
    elif choice == "4":
        example_monitoring()
    elif choice == "0":
        print("退出")
    else:
        print("无效的选择")
