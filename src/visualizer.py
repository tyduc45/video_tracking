"""
视频生成和可视化模块
"""

import cv2
import os
import logging
from typing import Tuple
import numpy as np

from pipeline_data import FrameData

logger = logging.getLogger(__name__)


class VideoGenerator:
    """从帧生成输出视频"""

    def __init__(self, fps: float = 30.0, codec: str = 'mp4v'):
        self.fps = fps
        self.codec = codec

    def generate_video(self, frame_dir: str, video_path: str,
                      resolution: Tuple[int, int]) -> bool:
        """从帧目录生成视频"""
        try:
            frame_files = sorted([
                f for f in os.listdir(frame_dir)
                if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))
            ])

            if not frame_files:
                logger.warning(f"No frames found in {frame_dir}")
                return False

            os.makedirs(os.path.dirname(video_path) or '.', exist_ok=True)

            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            writer = cv2.VideoWriter(video_path, fourcc, self.fps, resolution)

            if not writer.isOpened():
                logger.error(f"Failed to open video writer for {video_path}")
                return False

            frame_count = 0
            for frame_file in frame_files:
                frame_path = os.path.join(frame_dir, frame_file)
                frame = cv2.imread(frame_path)

                if frame is None:
                    logger.warning(f"Failed to read frame: {frame_path}")
                    continue

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
    """Pipeline输出处理器 - 保存结果帧和生成视频"""

    def __init__(self, output_dir: str = "result",
                 save_frames: bool = True,
                 save_video: bool = True,
                 draw_boxes: bool = True,
                 draw_ids: bool = True,
                 draw_confidence: bool = True,
                 fps: float = 30.0):
        self.output_dir = output_dir
        self.save_frames = save_frames
        self.save_video = save_video
        self.draw_boxes = draw_boxes
        self.draw_ids = draw_ids
        self.draw_confidence = draw_confidence
        self.fps = fps

        self.frame_info = {}
        self.video_generator = VideoGenerator(fps=fps)

    def process_frame(self, frame_data: FrameData):
        """处理一帧数据，可视化检测结果"""
        try:
            output_frame = frame_data.frame.copy()

            if frame_data.detections and self.draw_boxes:
                output_frame = self._draw_detections(output_frame, frame_data.detections)

            frame_info_text = f"Frame: {frame_data.frame_id}"
            cv2.putText(output_frame, frame_info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if self.save_frames:
                self._save_frame(frame_data, output_frame)

            if self.save_video:
                self._record_frame_info(frame_data, output_frame)

        except Exception as e:
            logger.error(f"Error processing frame: {e}")

    def _draw_detections(self, frame: np.ndarray, detections) -> np.ndarray:
        """绘制检测结果"""
        if hasattr(detections, 'plot'):
            return detections.plot()

        if isinstance(detections, list):
            for det in detections:
                if not isinstance(det, dict):
                    continue

                bbox = det.get('bbox', [])
                if len(bbox) < 4:
                    continue

                x1, y1, x2, y2 = bbox[:4]
                class_name = det.get('class_name', 'unknown')
                confidence = det.get('confidence', 0)
                track_id = det.get('track_id', None)

                color = (0, 255, 0)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                label = f"{class_name}"
                if self.draw_confidence:
                    label += f" {confidence:.2f}"
                if self.draw_ids and track_id is not None:
                    label += f" ID:{track_id}"

                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame,
                            (int(x1), int(y1) - text_height - 5),
                            (int(x1) + text_width, int(y1)),
                            color, -1)

                cv2.putText(frame, label, (int(x1), int(y1) - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return frame

    def _save_frame(self, frame_data: FrameData, output_frame: np.ndarray):
        """保存单个帧"""
        try:
            video_id = frame_data.video_id
            frame_id = frame_data.frame_id

            frame_dir = os.path.join(self.output_dir, video_id, "frames")
            os.makedirs(frame_dir, exist_ok=True)

            frame_path = os.path.join(frame_dir, f"frame_{frame_id:06d}.jpg")
            cv2.imwrite(frame_path, output_frame)

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

            temp_frame_dir = os.path.join(self.output_dir, video_id, "temp_frames")
            os.makedirs(temp_frame_dir, exist_ok=True)

            for frame_info in frames:
                frame_id = frame_info['frame_id']
                frame_data = frame_info['data']
                frame_path = os.path.join(temp_frame_dir, f"frame_{frame_id:06d}.jpg")
                cv2.imwrite(frame_path, frame_data)

            video_output_path = os.path.join(self.output_dir, video_id, f"{video_id}_tracked.mp4")
            self.video_generator.generate_video(temp_frame_dir, video_output_path, resolution)

            import shutil
            try:
                shutil.rmtree(temp_frame_dir)
            except:
                pass

        logger.info("Video generation completed")


def create_output_handler(config) -> PipelineOutputHandler:
    """根据配置创建输出处理器"""
    return PipelineOutputHandler(
        output_dir=config.output_dir,
        save_frames=config.save_frames,
        save_video=config.save_video,
        draw_boxes=config.draw_boxes,
        draw_ids=config.draw_ids,
        draw_confidence=config.draw_confidence,
        fps=config.save_fps,
    )
