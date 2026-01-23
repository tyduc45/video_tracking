#!/usr/bin/env python3
"""
使用示例
展示如何使用Pipeline系统的各种常见场景
"""

import logging
import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline_manager import PipelineManager
from video_source import LocalVideoSource, WebcamSource
from inference import YOLOInferencer, ByteTracker, ResultVisualizer
from visualizer import PipelineOutputHandler
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example1_single_video():
    """
    示例1：处理单个本地视频文件
    """
    logger.info("\n=== Example 1: Single Video Processing ===\n")
    
    # 创建管理器
    manager = PipelineManager(output_dir="result/example1")
    
    # 创建视频源
    video_source = LocalVideoSource("videos/video1.mp4")
    
    # 创建简单的推理函数（占位符）
    def inference_func(frame):
        # 实现YOLO推理
        return []
    
    # 创建追踪器
    tracker = ByteTracker()
    
    # 创建输出处理器
    output_handler = PipelineOutputHandler(output_dir="result/example1")
    
    def save_func(frame_data, output_dir):
        output_handler.process_frame(frame_data)
    
    # 创建Pipeline
    pipeline_id = manager.create_pipeline(
        video_source=video_source,
        inference_func=inference_func,
        tracker_instance=tracker,
        save_func=save_func
    )
    
    # 启动和等待
    manager.start_all()
    manager.wait_all(timeout=600)
    
    # 生成视频和统计
    output_handler.generate_all_videos()
    manager.print_all_statistics()


def example2_multiple_videos():
    """
    示例2：并行处理多个视频
    """
    logger.info("\n=== Example 2: Multiple Videos Processing ===\n")
    
    manager = PipelineManager(output_dir="result/example2", max_pipelines=5)
    
    # 创建多个视频源
    video_sources = [
        LocalVideoSource("videos/video1.mp4"),
        LocalVideoSource("videos/video2.mp4"),
    ]
    
    # 共享的推理和追踪
    def inference_func(frame):
        return []
    
    tracker = ByteTracker()
    output_handler = PipelineOutputHandler(output_dir="result/example2")
    
    def save_func(frame_data, output_dir):
        output_handler.process_frame(frame_data)
    
    # 为每个视频创建Pipeline
    for i, source in enumerate(video_sources):
        manager.create_pipeline(
            video_source=source,
            inference_func=inference_func,
            tracker_instance=tracker,
            save_func=save_func,
            pipeline_id=f"video_{i}"
        )
    
    # 并行处理
    manager.start_all()
    manager.wait_all(timeout=600)
    
    output_handler.generate_all_videos()
    manager.print_all_statistics()


def example3_mixed_sources():
    """
    示例3：混合处理本地视频和网络直播
    """
    logger.info("\n=== Example 3: Mixed Sources (Local + Webcam) ===\n")
    
    manager = PipelineManager(output_dir="result/example3")
    
    # 本地视频源
    local_source = LocalVideoSource("videos/video1.mp4")
    
    # 网络摄像头（RTSP）
    # webcam_source = WebcamSource("rtsp://192.168.1.100/stream")
    
    # 或本地摄像头
    # webcam_source = WebcamSource(0)
    
    def inference_func(frame):
        return []
    
    tracker = ByteTracker()
    output_handler = PipelineOutputHandler(output_dir="result/example3")
    
    def save_func(frame_data, output_dir):
        output_handler.process_frame(frame_data)
    
    # 创建Pipeline
    manager.create_pipeline(
        video_source=local_source,
        inference_func=inference_func,
        tracker_instance=tracker,
        save_func=save_func,
        pipeline_id="local_video"
    )
    
    # 只在有摄像头时添加
    # manager.create_pipeline(
    #     video_source=webcam_source,
    #     inference_func=inference_func,
    #     tracker_instance=tracker,
    #     save_func=save_func,
    #     pipeline_id="webcam"
    # )
    
    manager.start_all()
    manager.wait_all(timeout=600)
    
    output_handler.generate_all_videos()
    manager.print_all_statistics()


def example4_custom_inference():
    """
    示例4：使用自定义推理函数
    """
    logger.info("\n=== Example 4: Custom Inference Function ===\n")
    
    manager = PipelineManager(output_dir="result/example4")
    
    video_source = LocalVideoSource("videos/video1.mp4")
    
    # 自定义推理函数示例
    def custom_inference_func(frame):
        """
        自定义推理函数
        
        Args:
            frame: BGR格式的图像
        
        Returns:
            检测结果列表
        """
        # 这里可以实现自己的推理逻辑
        # 例如：使用不同的模型、自定义后处理等
        
        detections = []
        
        # 示例：在图像中心检测到一个对象
        h, w = frame.shape[:2]
        detections.append({
            'class_id': 0,
            'class_name': 'person',
            'confidence': 0.95,
            'bbox': [w//4, h//4, 3*w//4, 3*h//4],
        })
        
        return detections
    
    tracker = ByteTracker()
    output_handler = PipelineOutputHandler(output_dir="result/example4")
    
    def save_func(frame_data, output_dir):
        output_handler.process_frame(frame_data)
    
    manager.create_pipeline(
        video_source=video_source,
        inference_func=custom_inference_func,
        tracker_instance=tracker,
        save_func=save_func
    )
    
    manager.start_all()
    manager.wait_all(timeout=600)
    
    output_handler.generate_all_videos()
    manager.print_all_statistics()


def example5_realtime_monitoring():
    """
    示例5：实时监控多个摄像头（注意：需要实际的摄像头）
    """
    logger.info("\n=== Example 5: Real-time Multi-Camera Monitoring ===\n")
    
    manager = PipelineManager(output_dir="result/example5")
    
    # 模拟多个摄像头（实际使用时替换为真实URL）
    # camera_sources = [
    #     WebcamSource("rtsp://camera1.local/stream"),
    #     WebcamSource("rtsp://camera2.local/stream"),
    #     WebcamSource("rtsp://camera3.local/stream"),
    # ]
    
    # 演示用：使用本地视频模拟摄像头
    camera_sources = [
        LocalVideoSource("videos/video1.mp4"),
    ]
    
    def inference_func(frame):
        return []
    
    # 可以为每个摄像头配置不同的参数
    tracker = ByteTracker(track_high_thresh=0.7)
    output_handler = PipelineOutputHandler(
        output_dir="result/example5",
        save_frames=True,
        save_video=True,
        draw_boxes=True,
        draw_ids=True,
    )
    
    def save_func(frame_data, output_dir):
        output_handler.process_frame(frame_data)
    
    # 为每个摄像头创建独立的Pipeline
    for i, source in enumerate(camera_sources):
        manager.create_pipeline(
            video_source=source,
            inference_func=inference_func,
            tracker_instance=tracker,
            save_func=save_func,
            pipeline_id=f"camera_{i}"
        )
    
    # 并行监控所有摄像头
    manager.start_all()
    manager.wait_all(timeout=None)  # 无限等待，适合实时监控
    
    output_handler.generate_all_videos()
    manager.print_status()


def example6_performance_tuning():
    """
    示例6：性能优化
    """
    logger.info("\n=== Example 6: Performance Tuning ===\n")
    
    # 创建优化的配置
    from config import Config
    
    config = Config(
        model_path="model/yolo12n.pt",  # 使用较小的模型
        device="cuda",                    # 使用GPU
        confidence_threshold=0.6,         # 提高阈值减少检测数量
        batch_size=4,
        queue_size=20,                    # 增加队列大小
        max_pipelines=10,
    )
    
    manager = PipelineManager(
        output_dir="result/example6",
        max_pipelines=config.max_pipelines
    )
    
    video_source = LocalVideoSource("videos/video1.mp4")
    
    def inference_func(frame):
        return []
    
    tracker = ByteTracker()
    output_handler = PipelineOutputHandler(
        output_dir="result/example6",
        save_frames=False,  # 只保存视频，加快处理
        save_video=True,
        fps=config.save_fps,
    )
    
    def save_func(frame_data, output_dir):
        output_handler.process_frame(frame_data)
    
    manager.create_pipeline(
        video_source=video_source,
        inference_func=inference_func,
        tracker_instance=tracker,
        save_func=save_func
    )
    
    manager.start_all()
    manager.wait_all()
    
    output_handler.generate_all_videos()
    manager.print_all_statistics()


if __name__ == "__main__":
    # 选择要运行的示例
    examples = {
        '1': ('Single Video', example1_single_video),
        '2': ('Multiple Videos', example2_multiple_videos),
        '3': ('Mixed Sources', example3_mixed_sources),
        '4': ('Custom Inference', example4_custom_inference),
        '5': ('Real-time Monitoring', example5_realtime_monitoring),
        '6': ('Performance Tuning', example6_performance_tuning),
    }
    
    if len(sys.argv) > 1:
        example_id = sys.argv[1]
        if example_id in examples:
            name, func = examples[example_id]
            logger.info(f"Running Example {example_id}: {name}")
            func()
        else:
            logger.error(f"Unknown example: {example_id}")
            logger.info("Available examples:")
            for idx, (name, _) in examples.items():
                logger.info(f"  {idx}: {name}")
    else:
        logger.info("Usage: python examples.py <example_id>")
        logger.info("Available examples:")
        for idx, (name, _) in examples.items():
            logger.info(f"  {idx}: {name}")
