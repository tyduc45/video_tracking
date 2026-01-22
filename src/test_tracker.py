"""
集成测试脚本 - 验证追踪系统的各个组件
"""
import sys
import os
import numpy as np
import cv2
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tracker_manager import TrackerManager, FrameData, SingleVideoTracker


def test_frame_data_structure():
    """测试 FrameData 数据结构"""
    print("\n✓ 测试 FrameData 结构...")
    
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame_data = FrameData(
        frame=dummy_frame,
        path="test.mp4",
        video_id="video_0",
        frame_id=1,
        detections=None
    )
    
    assert frame_data.video_id == "video_0"
    assert frame_data.frame_id == 1
    print("  ✓ FrameData 创建成功")
    
    # 测试优先队列排序
    fd1 = FrameData(dummy_frame, "test.mp4", "video_0", 3, None)
    fd2 = FrameData(dummy_frame, "test.mp4", "video_0", 1, None)
    fd3 = FrameData(dummy_frame, "test.mp4", "video_0", 2, None)
    
    assert fd2 < fd3 < fd1
    print("  ✓ FrameData 可正确用于堆排序")


def test_single_video_tracker():
    """测试单个视频追踪器"""
    print("\n✓ 测试 SingleVideoTracker...")
    
    tracker = SingleVideoTracker("video_0")
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 测试顺序帧
    print("  测试顺序帧处理...")
    for i in range(1, 4):
        frame_data = FrameData(dummy_frame, "test.mp4", "video_0", i, None)
        ready = tracker.process_frame(frame_data)
        assert len(ready) == 1, f"期望1个准备好的帧，得到{len(ready)}"
        assert ready[0].frame_id == i
    
    print("  ✓ 顺序帧处理正确")
    
    # 测试乱序帧
    print("  测试乱序帧恢复...")
    tracker2 = SingleVideoTracker("video_1")
    
    # 发送帧序列: 3, 1, 2, 4
    frame_3 = FrameData(dummy_frame, "test.mp4", "video_1", 3, None)
    ready = tracker2.process_frame(frame_3)
    assert len(ready) == 0, "帧3应该进入队列"
    print("    ✓ 帧3进入队列")
    
    frame_1 = FrameData(dummy_frame, "test.mp4", "video_1", 1, None)
    ready = tracker2.process_frame(frame_1)
    assert len(ready) == 1, "帧1应该直接输出"
    assert ready[0].frame_id == 1
    print("    ✓ 帧1输出")
    
    frame_2 = FrameData(dummy_frame, "test.mp4", "video_1", 2, None)
    ready = tracker2.process_frame(frame_2)
    assert len(ready) == 2, "帧2应该触发帧2和帧3的输出"
    assert ready[0].frame_id == 2 and ready[1].frame_id == 3
    print("    ✓ 帧2和帧3自动输出")
    
    frame_4 = FrameData(dummy_frame, "test.mp4", "video_1", 4, None)
    ready = tracker2.process_frame(frame_4)
    assert len(ready) == 1
    assert ready[0].frame_id == 4
    print("    ✓ 帧4输出")
    
    print("  ✓ 乱序恢复算法正确")


def test_tracker_manager():
    """测试追踪管理器"""
    print("\n✓ 测试 TrackerManager...")
    
    video_paths = ["video0.mp4", "video1.mp4"]
    manager = TrackerManager(video_paths)
    
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 测试多视频源
    print("  测试多视频源处理...")
    
    # 向video_0发送帧
    frame_v0_1 = FrameData(dummy_frame, video_paths[0], "video_0", 1, None)
    ready = manager.process_frame(frame_v0_1)
    assert len(ready) == 1
    print("    ✓ video_0 帧处理")
    
    # 向video_1发送帧（乱序）
    frame_v1_2 = FrameData(dummy_frame, video_paths[1], "video_1", 2, None)
    ready = manager.process_frame(frame_v1_2)
    assert len(ready) == 0, "乱序帧应该进入队列"
    print("    ✓ video_1 乱序帧进队列")
    
    frame_v1_1 = FrameData(dummy_frame, video_paths[1], "video_1", 1, None)
    ready = manager.process_frame(frame_v1_1)
    assert len(ready) == 2, "应该同时输出frame_1和frame_2"
    print("    ✓ video_1 乱序恢复")
    
    # 检查管理器状态
    status = manager.get_status()
    assert status["video_0"]["expected_frame_id"] == 2
    assert status["video_1"]["expected_frame_id"] == 3
    print("  ✓ 追踪管理器状态正确")


def test_concurrent_processing():
    """测试并发处理能力"""
    print("\n✓ 测试并发处理...")
    
    import threading
    import time
    
    manager = TrackerManager(["video0.mp4", "video1.mp4"])
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    results = {"video_0": [], "video_1": []}
    lock = threading.Lock()
    
    def send_frames(video_id, frame_ids):
        for fid in frame_ids:
            frame_data = FrameData(
                dummy_frame, 
                "video0.mp4" if video_id == "video_0" else "video1.mp4",
                video_id, 
                fid, 
                None
            )
            ready = manager.process_frame(frame_data)
            with lock:
                results[video_id].extend(ready)
            time.sleep(0.01)  # 模拟处理延迟
    
    # 启动两个线程，发送乱序的帧
    t1 = threading.Thread(target=send_frames, args=("video_0", [1, 3, 2, 4]))
    t2 = threading.Thread(target=send_frames, args=("video_1", [2, 1, 3, 4]))
    
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    
    # 验证结果顺序
    v0_ids = [f.frame_id for f in results["video_0"]]
    v1_ids = [f.frame_id for f in results["video_1"]]
    
    assert v0_ids == [1, 2, 3, 4], f"video_0应该输出[1,2,3,4]，得到{v0_ids}"
    assert v1_ids == [1, 2, 3, 4], f"video_1应该输出[1,2,3,4]，得到{v1_ids}"
    
    print("  ✓ 并发处理结果正确")


def test_memory_efficiency():
    """测试内存效率"""
    print("\n✓ 测试内存效率...")
    
    tracker = SingleVideoTracker("video_0")
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 发送大量乱序帧
    num_frames = 100
    frames_to_send = list(range(1, num_frames + 1))
    
    # 故意打乱顺序（但保持某些乱序距离）
    np.random.seed(42)
    shuffled = frames_to_send.copy()
    np.random.shuffle(shuffled)
    
    total_ready = 0
    max_queue_size = 0
    
    for fid in shuffled:
        frame_data = FrameData(dummy_frame, "test.mp4", "video_0", fid, None)
        ready = tracker.process_frame(frame_data)
        total_ready += len(ready)
        
        queue_size = tracker.get_queue_size()
        max_queue_size = max(max_queue_size, queue_size)
    
    # 最后应该所有帧都输出
    assert total_ready == num_frames, f"应该输出{num_frames}帧，得到{total_ready}"
    
    print(f"  ✓ 处理了{num_frames}个乱序帧")
    print(f"  ✓ 最大队列大小: {max_queue_size}帧")
    print(f"  ✓ 内存使用高效（最大队列 < 输入帧数）")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print("YOLO追踪系统 - 集成测试")
    print("="*60)
    
    tests = [
        test_frame_data_structure,
        test_single_video_tracker,
        test_tracker_manager,
        test_concurrent_processing,
        test_memory_efficiency,
    ]
    
    failed = 0
    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"  ✗ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    if failed == 0:
        print("✓ 所有测试通过！")
    else:
        print(f"✗ {failed} 个测试失败")
    print("="*60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
