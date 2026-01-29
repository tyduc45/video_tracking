#!/usr/bin/env python3
"""
主程序入口
多视频批处理推理架构
"""

import logging
import sys
import os
from pathlib import Path
import argparse
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

from video_source import LocalVideoSource, WebcamSource
from config import load_config, save_config, Config
from inference import YOLOInferencer, PassthroughTracker, ByteTrackTracker
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
        for path in input_paths:
            if path.isdigit():
                sources.append(WebcamSource(int(path)))
            elif path.startswith(('http://', 'https://', 'rtsp://')):
                sources.append(WebcamSource(path))
            else:
                if os.path.exists(path):
                    sources.append(LocalVideoSource(path))
                else:
                    logger.warning(f"Input file not found: {path}")
    else:
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
        description='YOLO对象追踪系统 - 多视频批处理模式'
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
                       help='批处理大小（默认32）')
    parser.add_argument('--no-frames', action='store_true',
                       help='不保存帧')
    parser.add_argument('--no-video', action='store_true',
                       help='不生成视频')
    parser.add_argument('--display', action='store_true',
                       help='实时显示检测结果（按Q或ESC退出）')
    parser.add_argument('--display-width', type=int, default=1280,
                       help='显示窗口宽度（默认1280）')
    parser.add_argument('--display-height', type=int, default=720,
                       help='显示窗口高度（默认720）')

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

    # 7. 运行批处理模式
    return run_batch_mode(config, video_sources, args)


def run_batch_mode(config, video_sources, args):
    """
    批处理模式 - 多视频并行，共享批推理

    架构：
    Reader0 ──┐                              ┌──> Tracker0 (PassThrough) ──> Saver0
    Reader1 ──┼──> BatchProcessor ──────────┼──> Tracker1 (PassThrough) ──> Saver1
    Reader2 ──┤   (GPU批推理)                ├──> Tracker2 (PassThrough) ──> Saver2
    ...      ─┘                              └──> ...
    """
    from batch_inference_system import MultiVideoPipeline, calculate_k_values

    num_videos = len(video_sources)
    k_values = calculate_k_values(num_videos, args.batch_size)

    logger.info("=" * 60)
    logger.info("Running in BATCH MODE")
    logger.info(f"  Videos: {num_videos}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  k_values: {k_values}")
    logger.info("=" * 60)

    # 1. 加载推理模型
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
        batch_inference_func = inferencer.infer_batch
    except Exception as e:
        logger.error(f"Failed to load inference model: {e}")
        return 1

    # 2. 创建 ByteTrack 追踪器工厂
    def tracker_factory(pipeline_id: str):
        logger.info(f"Creating ByteTrackTracker for {pipeline_id}")
        return ByteTrackTracker(session_id=pipeline_id)

    # 3. 创建输出处理器（支持实时显示）
    realtime_display = getattr(args, 'display', False)
    display_width = getattr(args, 'display_width', 1280)
    display_height = getattr(args, 'display_height', 720)

    output_handler = create_output_handler(
        config,
        realtime_display=realtime_display,
        window_width=display_width,
        window_height=display_height
    )

    if realtime_display:
        logger.info(f"Realtime display enabled: {display_width}x{display_height}")
        logger.info("Press 'Q' or 'ESC' to stop")

    def save_func(frame_data, output_dir):
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
        queue_size=200
    )

    # 设置全局停止事件
    global _global_stop_event
    _global_stop_event = pipeline.stop_event

    # 5. 启动处理
    logger.info("Starting batch processing...")
    pipeline.start()

    # 6. 等待完成（支持实时显示退出）
    logger.info("Waiting for processing to complete...")
    user_quit = False
    try:
        if realtime_display:
            # 实时显示模式：检查显示窗口是否关闭
            import time
            while not pipeline.stop_event.is_set():
                if not output_handler.is_display_active():
                    logger.info("Display window closed, stopping...")
                    pipeline.stop()
                    user_quit = output_handler.is_user_quit()
                    break
                time.sleep(0.1)
            pipeline.wait(timeout=10.0)
        else:
            pipeline.wait(timeout=None)
    except KeyboardInterrupt:
        logger.info("Received interrupt, stopping...")
        pipeline.stop()
        user_quit = True
        pipeline.wait(timeout=10.0)
    finally:
        output_handler.stop_display()

    # 用户主动退出，直接终止程序
    if user_quit:
        logger.info("User quit, exiting program...")
        sys.exit(0)

    # 7. 生成输出视频
    if config.save_video:
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

    logger.info("Batch processing completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
