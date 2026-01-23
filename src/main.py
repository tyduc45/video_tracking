#!/usr/bin/env python3
"""
主程序入口
单一视频单一流水线架构
"""

import logging
import sys
import os
from pathlib import Path
import argparse
import threading
import signal

# 全局停止事件，用于 Ctrl+C 处理
_global_stop_event = None

def _signal_handler(signum, frame):
    """处理 Ctrl+C 信号"""
    print("\n收到中断信号，正在停止...")
    if _global_stop_event:
        _global_stop_event.set()
    sys.exit(1)

# 注册信号处理器
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

# 添加src目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline_manager import PipelineManager
from video_source import LocalVideoSource, WebcamSource
from config import load_config, save_config, Config
from inference import YOLOInferencer, ByteTracker
from visualizer import create_output_handler

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_logging(config: Config):
    """配置日志系统"""
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.log_level))
    
    if config.log_file:
        os.makedirs(os.path.dirname(config.log_file) or '.', exist_ok=True)
        file_handler = logging.FileHandler(config.log_file)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def create_video_sources(config: Config, input_paths: list = None):
    """
    创建视频源列表
    
    Args:
        config: 配置对象
        input_paths: 输入路径列表（若None则自动发现）
    
    Returns:
        VideoSource列表
    """
    sources = []
    
    if input_paths:
        # 使用指定的输入路径
        for path in input_paths:
            if path.isdigit():
                # 本地摄像头
                sources.append(WebcamSource(int(path)))
            elif path.startswith(('http://', 'https://', 'rtsp://')):
                # 网络流
                sources.append(WebcamSource(path))
            else:
                # 本地文件
                if os.path.exists(path):
                    sources.append(LocalVideoSource(path))
                else:
                    logger.warning(f"Input file not found: {path}")
    else:
        # 自动发现输入目录中的视频文件
        input_dir = config.input_dir
        if os.path.isdir(input_dir):
            for file in os.listdir(input_dir):
                if file.lower().endswith(('.mp4', '.avi', '.mkv', '.mov', '.flv')):
                    file_path = os.path.join(input_dir, file)
                    sources.append(LocalVideoSource(file_path))
    
    if not sources:
        logger.warning(f"No video sources found in {config.input_dir}")
    
    return sources


def main():
    """主程序"""
    parser = argparse.ArgumentParser(
        description='YOLO对象追踪系统 - 支持批处理和单视频模式'
    )
    parser.add_argument('-c', '--config', type=str, default=None,
                       help='配置文件路径')
    parser.add_argument('-i', '--input', type=str, nargs='+', default=None,
                       help='输入视频文件或URL列表')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='输出目录')
    parser.add_argument('-m', '--model', type=str, default=None,
                       help='YOLO模型路径')
    parser.add_argument('-d', '--device', type=str, default=None,
                       choices=['cpu', 'cuda'], help='推理设备')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='批处理大小（多视频模式，默认32）')
    parser.add_argument('--use-traditional', action='store_true',
                       help='使用传统单视频模式（不使用批处理）')
    parser.add_argument('--no-frames', action='store_true',
                       help='不保存帧')
    parser.add_argument('--no-video', action='store_true',
                       help='不生成视频')
    
    args = parser.parse_args()
    
    # 1. 加载或创建配置
    logger.info("Loading configuration...")
    config = load_config(args.config)
    
    # 2. 覆盖命令行参数
    if args.output:
        config.output_dir = args.output
    if args.model:
        config.model_path = args.model
    if args.device:
        config.device = args.device
    if args.no_frames:
        config.save_frames = False
    if args.no_video:
        config.save_video = False
    
    # 3. 验证配置
    check_input_dir = (args.input is None)
    if not config.validate(check_input_dir=check_input_dir):
        logger.error("Configuration validation failed")
        return 1
    
    # 4. 设置日志
    setup_logging(config)
    logger.info(f"Configuration: {config.to_dict()}")
    
    # 5. 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 6. 创建视频源
    logger.info("Creating video sources...")
    video_sources = create_video_sources(config, args.input)
    
    if not video_sources:
        logger.error("No video sources found")
        return 1
    
    logger.info(f"Found {len(video_sources)} video source(s)")
    
    # 7. 决定使用批处理模式还是传统模式
    use_batch_mode = len(video_sources) > 1 and not args.use_traditional
    
    if use_batch_mode:
        return run_batch_mode(config, video_sources, args)
    else:
        return run_traditional_mode(config, video_sources, args)


def run_batch_mode(config, video_sources, args):
    """
    批处理模式 - 多视频并行，共享批推理
    
    架构：
    Reader0 ──┐                              ┌──> Tracker0 (PassThrough) ──> Saver0
    Reader1 ──┼──> BatchProcessor ──────────┼──> Tracker1 (PassThrough) ──> Saver1
    Reader2 ──┤   (GPU批推理)                ├──> Tracker2 (PassThrough) ──> Saver2
    ...      ─┘                              └──> ...
    
    注意：BatchProcessor 已经完成了检测推理，Tracker 只是透传结果。
    如需追踪ID，可使用纯ByteTrack库（不含YOLO模型）。
    
    k值计算（每个视频可打进批次的帧数）:
    - 基础k = batch_size // num_videos (向下取整)
    - 最后一个视频补齐剩余
    - 例如 n=15, batch_size=32: k_values=[2,2,2,2,2,2,2,2,2,2,2,2,2,2,4]
    """
    from batch_inference_system import MultiVideoPipeline, calculate_k_values
    from inference import PassthroughTracker
    
    num_videos = len(video_sources)
    k_values = calculate_k_values(num_videos, args.batch_size)
    
    logger.info("=" * 60)
    logger.info("Running in BATCH MODE (架构革新)")
    logger.info(f"  Videos: {num_videos}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  k_values: {k_values}")
    logger.info("=" * 60)
    
    # 1. 加载推理模型（使用批推理函数）
    logger.info("Loading inference model for batch processing...")
    try:
        inferencer = YOLOInferencer(
            model_path=config.model_path,
            model_dir=config.model_dir,
            device=config.device,
            use_half=config.use_half,
            confidence_threshold=config.confidence_threshold,
            iou_threshold=config.iou_threshold,
            batch_size=args.batch_size
        )
        # 使用批推理函数
        batch_inference_func = inferencer.infer_batch
    except Exception as e:
        logger.error(f"Failed to load inference model: {e}")
        return 1
    
    # 2. 创建透传追踪器工厂
    # 批处理模式下，推理已由 BatchProcessor 完成
    # PassthroughTracker 只透传数据，不加载额外的 YOLO 模型
    # 这样可以避免 TensorRT 多实例 CUDA 冲突
    def tracker_factory(pipeline_id: str):
        """
        为每个视频创建透传追踪器
        
        注意：如果需要追踪ID，可以集成纯ByteTrack库（如supervision）
        """
        logger.info(f"Creating PassthroughTracker for {pipeline_id}")
        return PassthroughTracker(session_id=pipeline_id)
    
    # 3. 创建输出处理器
    output_handler = create_output_handler(config)
    
    def save_func(frame_data, output_dir):
        """保存函数包装"""
        output_handler.process_frame(frame_data)
    
    # 4. 创建批处理流水线
    logger.info("Creating batch processing pipeline...")
    pipeline = MultiVideoPipeline(
        video_sources=video_sources,
        inference_func=batch_inference_func,
        tracker_factory=tracker_factory,
        save_func=save_func,
        output_dir=config.output_dir,
        batch_size=args.batch_size,
        queue_size=200  # 增加队列大小，防止生产者阻塞
    )
    
    # 设置全局停止事件，用于 Ctrl+C 处理
    global _global_stop_event
    _global_stop_event = pipeline.stop_event
    
    # 5. 启动处理
    logger.info("Starting batch processing...")
    pipeline.start()
    
    # 6. 等待完成
    logger.info("Waiting for processing to complete...")
    try:
        success = pipeline.wait(timeout=None)
    except KeyboardInterrupt:
        logger.info("Received interrupt, stopping...")
        pipeline.stop()
        success = pipeline.wait(timeout=10.0)
    
    # 7. 生成输出视频
    logger.info("Generating output videos...")
    output_handler.generate_all_videos()
    
    # 8. 打印统计
    stats = pipeline.get_stats()
    logger.info("=" * 60)
    logger.info("Batch Processing Statistics:")
    logger.info(f"  k_values: {stats.get('k_values', [])}")
    logger.info(f"  Total batches: {stats['batch_processor'].get('total_batches', 0)}")
    logger.info(f"  Total frames: {stats['batch_processor'].get('total_frames', 0)}")
    logger.info(f"  Inference time: {stats['batch_processor'].get('inference_time', 0):.2f}s")
    logger.info("=" * 60)
    
    # 9. 保存配置
    config_output_path = os.path.join(config.output_dir, "config.json")
    save_config(config, config_output_path)
    
    if success:
        logger.info("✓ Batch processing completed successfully")
        return 0
    else:
        logger.warning("✗ Batch processing did not complete normally")
        return 1


def run_traditional_mode(config, video_sources, args):
    """
    传统模式 - 单视频独立处理
    """
    logger.info("=" * 60)
    logger.info("Running in TRADITIONAL MODE")
    logger.info(f"  Videos: {len(video_sources)}")
    logger.info("=" * 60)
    
    # 1. 创建Pipeline管理器
    manager = PipelineManager(
        output_dir=config.output_dir,
        max_pipelines=config.max_pipelines
    )
    
    # 2. 加载推理模型
    logger.info("Loading inference model...")
    try:
        inferencer = YOLOInferencer(
            model_path=config.model_path,
            model_dir=config.model_dir,
            device=config.device,
            use_half=config.use_half,
            confidence_threshold=config.confidence_threshold,
            iou_threshold=config.iou_threshold,
            batch_size=config.inference_batch_size
        )
        inference_func = inferencer.infer
    except Exception as e:
        logger.error(f"Failed to load inference model: {e}")
        return 1
    
    # 3. 加载追踪器
    logger.info("Loading tracker...")
    try:
        tracker = ByteTracker(
            model_path=config.model_path,
            device=config.device,
            track_high_thresh=config.track_high_thresh,
            track_low_thresh=config.track_low_thresh,
            track_buffer=config.track_buffer,
            frame_rate=30.0
        )
    except Exception as e:
        logger.error(f"Failed to load tracker: {e}")
        return 1
    
    # 4. 创建输出处理器
    output_handler = create_output_handler(config)
    
    def save_func(frame_data, output_dir):
        output_handler.process_frame(frame_data)
    
    # 5. 创建Pipeline
    logger.info("Creating pipelines...")
    for i, source in enumerate(video_sources):
        pipeline_id = f"video_{i}"
        try:
            manager.create_pipeline(
                video_source=source,
                inference_func=inference_func,
                tracker_instance=tracker,
                save_func=save_func,
                pipeline_id=pipeline_id
            )
        except Exception as e:
            logger.error(f"Failed to create pipeline {pipeline_id}: {e}")
    
    # 6. 启动
    logger.info("Starting all pipelines...")
    manager.start_all()
    
    # 7. 等待完成
    logger.info("Waiting for all pipelines to complete...")
    success = manager.wait_all(timeout=None)
    
    # 8. 生成视频
    logger.info("Generating output videos...")
    output_handler.generate_all_videos()
    
    # 9. 打印统计
    manager.print_status()
    manager.print_all_statistics()
    
    # 10. 保存配置
    config_output_path = os.path.join(config.output_dir, "config.json")
    save_config(config, config_output_path)
    
    if success:
        logger.info("✓ All pipelines completed successfully")
        return 0
    else:
        logger.warning("✗ Some pipelines did not complete")
        return 1


if __name__ == "__main__":
    sys.exit(main())
