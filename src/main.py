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
        description='单一视频单一流水线的YOLO对象追踪系统'
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
    # 如果指定了-i参数，则不检查input_dir
    check_input_dir = (args.input is None)
    if not config.validate(check_input_dir=check_input_dir):
        logger.error("Configuration validation failed")
        return 1
    
    # 4. 设置日志
    setup_logging(config)
    logger.info(f"Configuration: {config.to_dict()}")
    
    # 5. 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 6. 创建Pipeline管理器
    logger.info("Initializing PipelineManager...")
    manager = PipelineManager(
        output_dir=config.output_dir,
        max_pipelines=config.max_pipelines
    )
    
    # 7. 创建视频源
    logger.info("Creating video sources...")
    video_sources = create_video_sources(config, args.input)
    
    if not video_sources:
        logger.error("No video sources found")
        return 1
    
    logger.info(f"Found {len(video_sources)} video source(s)")
    
    # 8. 加载推理模型和追踪器
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
    
    logger.info("Loading tracker...")
    try:
        # ByteTracker 需要模型路径用于YOLO原生track方法
        tracker = ByteTracker(
            model_path=config.model_path,
            device=config.device,
            track_high_thresh=config.track_high_thresh,
            track_low_thresh=config.track_low_thresh,
            track_buffer=config.track_buffer,
            frame_rate=30.0  # 可从视频源获取
        )
    except Exception as e:
        logger.error(f"Failed to load tracker: {e}")
        return 1
    
    # 9. 创建输出处理器
    output_handler = create_output_handler(config)
    
    def save_func(frame_data, output_dir):
        """保存函数包装"""
        output_handler.process_frame(frame_data)
    
    # 10. 创建Pipeline
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
    
    # 11. 启动所有Pipeline
    logger.info("Starting all pipelines...")
    manager.start_all()
    
    # 12. 等待所有Pipeline完成
    logger.info("Waiting for all pipelines to complete...")
    success = manager.wait_all(timeout=None)  # 无限等待
    
    # 13. 生成最终视频
    logger.info("Generating output videos...")
    output_handler.generate_all_videos()
    
    # 14. 打印统计信息
    manager.print_status()
    manager.print_all_statistics()
    
    # 15. 保存配置
    config_output_path = os.path.join(config.output_dir, "config.json")
    save_config(config, config_output_path)
    logger.info(f"Configuration saved to {config_output_path}")
    
    # 16. 完成
    if success:
        logger.info("✓ All pipelines completed successfully")
        return 0
    else:
        logger.warning("✗ Some pipelines did not complete")
        return 1


if __name__ == "__main__":
    sys.exit(main())
