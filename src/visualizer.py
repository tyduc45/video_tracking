"""
视频生成和可视化模块
"""

import cv2
import os
import logging
from typing import List, Tuple
from pathlib import Path
import numpy as np

from pipeline_data import FrameData
from inference import ResultVisualizer

logger = logging.getLogger(__name__)


class VideoGenerator:
    """
    从帧生成输出视频
    """
    
    def __init__(self, fps: float = 30.0, codec: str = 'mp4v'):
        """
        Args:
            fps: 输出视频帧率
            codec: 编码格式（mp4v, h264等）
        """
        self.fps = fps
        self.codec = codec
    
    def generate_video(self, frame_dir: str, video_path: str,
                      resolution: Tuple[int, int]) -> bool:
        """
        从帧目录生成视频
        
        Args:
            frame_dir: 帧所在目录
            video_path: 输出视频路径
            resolution: 视频分辨率 (width, height)
        
        Returns:
            是否成功
        """
        try:
            # 获取所有帧文件
            frame_files = sorted([
                f for f in os.listdir(frame_dir)
                if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))
            ])
            
            if not frame_files:
                logger.warning(f"No frames found in {frame_dir}")
                return False
            
            # 创建输出目录
            os.makedirs(os.path.dirname(video_path) or '.', exist_ok=True)
            
            # 初始化视频写入器
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            writer = cv2.VideoWriter(video_path, fourcc, self.fps, resolution)
            
            if not writer.isOpened():
                logger.error(f"Failed to open video writer for {video_path}")
                return False
            
            # 逐帧写入
            frame_count = 0
            for frame_file in frame_files:
                frame_path = os.path.join(frame_dir, frame_file)
                frame = cv2.imread(frame_path)
                
                if frame is None:
                    logger.warning(f"Failed to read frame: {frame_path}")
                    continue
                
                # 确保帧大小正确
                if frame.shape[:2][::-1] != resolution:
                    frame = cv2.resize(frame, resolution)
                
                writer.write(frame)
                frame_count += 1
            
            writer.release()
            
            logger.info(f"Generated video: {video_path} ({frame_count} frames)")
            return True
        
        except Exception as e:
            logger.error(f"Error generating video: {e}")
            return False


class PipelineOutputHandler:
    """
    Pipeline输出处理器
    负责保存结果帧和生成视频
    """
    
    def __init__(self, output_dir: str = "result",
                 save_frames: bool = True,
                 save_video: bool = True,
                 draw_boxes: bool = True,
                 draw_ids: bool = True,
                 draw_confidence: bool = True,
                 fps: float = 30.0):
        """
        Args:
            output_dir: 输出根目录
            save_frames: 是否保存帧
            save_video: 是否生成视频
            draw_boxes: 是否绘制检测框
            draw_ids: 是否绘制追踪ID
            draw_confidence: 是否绘制置信度
            fps: 输出视频帧率
        """
        self.output_dir = output_dir
        self.save_frames = save_frames
        self.save_video = save_video
        self.draw_boxes = draw_boxes
        self.draw_ids = draw_ids
        self.draw_confidence = draw_confidence
        self.fps = fps
        
        self.frame_info = {}  # video_id -> {'frames': [], 'fps': fps, 'resolution': (w, h)}
        self.video_generator = VideoGenerator(fps=fps)
    
    def process_frame(self, frame_data: FrameData):
        """
        处理一帧数据，使用YOLO原生plot方法可视化
        
        Args:
            frame_data: Pipeline传递的帧数据
        """
        try:
            output_frame = frame_data.frame.copy()
            
            # 使用 YOLO 原生的 plot() 方法绘制检测框和追踪ID
            # 这个方法会自动处理 boxes 和 track_id 的绘制，确保帧间一致性
            if frame_data.detections and self.draw_boxes:
                output_frame = frame_data.detections.plot()
            
            # 添加帧信息
            frame_info_text = f"Frame: {frame_data.frame_id}"
            cv2.putText(output_frame, frame_info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 保存帧
            if self.save_frames:
                self._save_frame(frame_data, output_frame)
            
            # 记录帧信息用于生成视频
            if self.save_video:
                self._record_frame_info(frame_data, output_frame)
        
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
    
    def _save_frame(self, frame_data: FrameData, output_frame: np.ndarray):
        """保存单个帧"""
        try:
            video_id = frame_data.video_id
            frame_id = frame_data.frame_id
            
            # 创建视频专用目录
            frame_dir = os.path.join(self.output_dir, video_id, "frames")
            os.makedirs(frame_dir, exist_ok=True)
            
            # 保存帧
            frame_path = os.path.join(frame_dir, f"frame_{frame_id:06d}.jpg")
            ResultVisualizer.save_frame(output_frame, frame_path)
        
        except Exception as e:
            logger.warning(f"Failed to save frame: {e}")
    
    def _record_frame_info(self, frame_data: FrameData, output_frame: np.ndarray):
        """记录帧信息用于生成视频"""
        video_id = frame_data.video_id
        
        if video_id not in self.frame_info:
            self.frame_info[video_id] = {
                'frames': [],
                'fps': self.fps,
                'resolution': (output_frame.shape[1], output_frame.shape[0]),
            }
        
        self.frame_info[video_id]['frames'].append({
            'frame_id': frame_data.frame_id,
            'data': output_frame.copy(),
        })
    
    def generate_all_videos(self):
        """生成所有Pipeline的输出视频"""
        if not self.save_video:
            logger.info("Video generation disabled")
            return
        
        logger.info("Generating output videos...")
        
        for video_id, info in self.frame_info.items():
            frames = info['frames']
            fps = info['fps']
            resolution = info['resolution']
            
            if not frames:
                logger.warning(f"No frames recorded for {video_id}")
                continue
            
            # 创建临时帧目录
            temp_frame_dir = os.path.join(self.output_dir, video_id, "temp_frames")
            os.makedirs(temp_frame_dir, exist_ok=True)
            
            # 保存所有帧到临时目录
            for frame_info in frames:
                frame_id = frame_info['frame_id']
                frame_data = frame_info['data']
                frame_path = os.path.join(temp_frame_dir, f"frame_{frame_id:06d}.jpg")
                ResultVisualizer.save_frame(frame_data, frame_path)
            
            # 生成视频
            video_output_path = os.path.join(self.output_dir, video_id, f"{video_id}_tracked.mp4")
            self.video_generator.generate_video(temp_frame_dir, video_output_path, resolution)
            
            # 清理临时文件
            import shutil
            try:
                shutil.rmtree(temp_frame_dir)
            except:
                pass
        
        logger.info("Video generation completed")


def create_output_handler(config) -> PipelineOutputHandler:
    """
    根据配置创建输出处理器
    
    Args:
        config: Config对象
    
    Returns:
        PipelineOutputHandler实例
    """
    return PipelineOutputHandler(
        output_dir=config.output_dir,
        save_frames=config.save_frames,
        save_video=config.save_video,
        draw_boxes=config.draw_boxes,
        draw_ids=config.draw_ids,
        draw_confidence=config.draw_confidence,
        fps=config.save_fps,
    )
