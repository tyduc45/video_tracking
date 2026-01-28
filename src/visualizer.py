"""
视频生成和可视化模块
"""

import cv2
import os
import logging
import threading
from typing import Tuple, Optional
from queue import Queue, Empty
import numpy as np

from pipeline_data import FrameData

logger = logging.getLogger(__name__)


class RealtimeDisplay:
    """实时显示器 - 在窗口中实时显示处理结果"""

    def __init__(self, window_name: str = "Video Object Detection",
                 display_fps: float = 30.0,
                 window_width: int = 1280,
                 window_height: int = 720):
        self.window_name = window_name
        self.display_fps = display_fps
        self.window_width = window_width
        self.window_height = window_height
        self.frame_interval = 1.0 / display_fps

        self.frame_queue: Queue = Queue(maxsize=100)
        self.stop_event = threading.Event()
        self.user_quit = threading.Event()  # 用户主动关闭窗口
        self.display_thread: Optional[threading.Thread] = None
        self.is_running = False

        self.current_frames = {}  # video_id -> latest frame
        self.lock = threading.Lock()

    def start(self):
        """启动显示线程"""
        if self.is_running:
            return

        self.is_running = True
        self.stop_event.clear()
        self.display_thread = threading.Thread(
            target=self._display_loop,
            name="RealtimeDisplay",
            daemon=True
        )
        self.display_thread.start()
        logger.info(f"RealtimeDisplay started: {self.window_name}")

    def stop(self):
        """停止显示"""
        self.stop_event.set()
        if self.display_thread and self.display_thread.is_alive():
            self.display_thread.join(timeout=2.0)
        cv2.destroyAllWindows()
        self.is_running = False
        logger.info("RealtimeDisplay stopped")

    def update_frame(self, video_id: str, frame: np.ndarray):
        """更新某个视频的最新帧"""
        with self.lock:
            self.current_frames[video_id] = frame.copy()

    def _display_loop(self):
        """显示循环"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        import time
        last_time = time.time()
        window_resized = False

        while not self.stop_event.is_set():
            current_time = time.time()
            elapsed = current_time - last_time

            if elapsed < self.frame_interval:
                time.sleep(0.001)
                continue

            last_time = current_time

            with self.lock:
                if not self.current_frames:
                    continue

                # 拼接所有视频的帧
                combined = self._combine_frames(self.current_frames)

            if combined is not None:
                # 首次显示时，根据画布大小调整窗口
                if not window_resized:
                    h, w = combined.shape[:2]
                    # 限制最大窗口大小，避免超出屏幕
                    max_win_w, max_win_h = 1920, 1080
                    scale = min(1.0, max_win_w / w, max_win_h / h)
                    win_w = int(w * scale)
                    win_h = int(h * scale)
                    cv2.resizeWindow(self.window_name, win_w, win_h)
                    window_resized = True

                cv2.imshow(self.window_name, combined)

            key = cv2.waitKey(1) & 0xFF

            # 检查窗口是否被关闭（点击X按钮）
            try:
                window_visible = cv2.getWindowProperty(
                    self.window_name, cv2.WND_PROP_VISIBLE
                )
                if window_visible < 1:
                    logger.info("Display window closed by user (X button)")
                    self.user_quit.set()
                    self.stop_event.set()
                    break
            except cv2.error:
                # 窗口已销毁
                self.user_quit.set()
                self.stop_event.set()
                break

            # 检查按键退出
            if key == ord('q') or key == 27:  # q or ESC
                logger.info("Display closed by user (keyboard)")
                self.user_quit.set()
                self.stop_event.set()
                break

        cv2.destroyAllWindows()

    def _combine_frames(self, frames: dict) -> Optional[np.ndarray]:
        """
        将多个视频帧拼接成一个显示画面
        - 保留各自原始分辨率（在单元格内）
        - 小分辨率向大分辨率对齐
        - 画面垂直水平居中
        - 空白区域用灰色填充
        """
        if not frames:
            return None

        # 按video_id排序，保证显示顺序一致
        sorted_items = sorted(frames.items(), key=lambda x: x[0])
        frame_list = [item[1] for item in sorted_items]
        n = len(frame_list)

        if n == 1:
            return self._center_frame_on_canvas(frame_list[0])

        if n == 2:
            # 两个视频：水平并排，保留原始分辨率，垂直居中对齐
            return self._combine_two_frames(frame_list[0], frame_list[1])

        # 多于2个视频：使用网格布局
        return self._combine_grid_frames(frame_list)

    def _combine_two_frames(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """
        并排显示两个视频
        - 保留各自原始分辨率
        - 垂直居中对齐
        - 灰色填充空白区域
        """
        h1, w1 = frame1.shape[:2]
        h2, w2 = frame2.shape[:2]

        # 最大高度
        max_height = max(h1, h2)
        max_width = max(w1,w2)
        total_width = 2 * max_width

        # 创建灰色画布
        canvas = np.full((max_height, total_width, 3), 128, dtype=np.uint8)

        # 放置第一个帧（垂直水平居中）
        y1_offset = (max_height - h1) // 2
        x1_offset = (max_width - w1) // 2
        canvas[y1_offset:y1_offset + h1, x1_offset:x1_offset+w1] = frame1

        # 放置第二个帧（垂直水平居中）
        y2_offset = (max_height - h2) // 2
        x2_offset = (max_width - w2) // 2
        canvas[y2_offset:y2_offset + h2, max_width+x2_offset:max_width + x2_offset + w2] = frame2

        return canvas

    def _combine_grid_frames(self, frame_list: list) -> np.ndarray:
        """
        网格布局显示多个视频
        - 每个单元格内保留原始宽高比
        - 居中显示，灰色填充
        """
        n = len(frame_list)

        # 计算网格布局
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))

        # 找到最大宽度和高度
        max_w = max(f.shape[1] for f in frame_list)
        max_h = max(f.shape[0] for f in frame_list)

        # 创建灰色画布
        canvas_h = max_h * rows
        canvas_w = max_w * cols
        canvas = np.full((canvas_h, canvas_w, 3), 128, dtype=np.uint8)

        for i, frame in enumerate(frame_list):
            row = i // cols
            col = i % cols

            h, w = frame.shape[:2]

            # 计算该帧在单元格中的居中位置
            cell_x = col * max_w
            cell_y = row * max_h

            x_offset = cell_x + (max_w - w) // 2
            y_offset = cell_y + (max_h - h) // 2

            canvas[y_offset:y_offset + h, x_offset:x_offset + w] = frame

        return canvas

    def _center_frame_on_canvas(self, frame: np.ndarray) -> np.ndarray:
        """将单帧居中放置在灰色画布上"""
        h, w = frame.shape[:2]

        # 计算缩放比例（保持宽高比）
        scale_w = self.window_width / w
        scale_h = self.window_height / h
        scale = min(scale_w, scale_h)

        # 如果帧比窗口小，不放大，直接居中
        if scale > 1.0:
            scale = 1.0

        new_w = int(w * scale)
        new_h = int(h * scale)

        if scale < 1.0:
            resized = cv2.resize(frame, (new_w, new_h))
        else:
            resized = frame

        # 创建灰色画布
        canvas = np.full((self.window_height, self.window_width, 3), 128, dtype=np.uint8)

        # 居中放置
        x_offset = (self.window_width - new_w) // 2
        y_offset = (self.window_height - new_h) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        return canvas


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
    """Pipeline输出处理器 - 保存结果帧、生成视频、实时显示"""

    def __init__(self, output_dir: str = "result",
                 save_frames: bool = True,
                 save_video: bool = True,
                 draw_boxes: bool = True,
                 draw_ids: bool = True,
                 draw_confidence: bool = True,
                 fps: float = 30.0,
                 realtime_display: bool = False,
                 window_width: int = 1280,
                 window_height: int = 720):
        self.output_dir = output_dir
        self.save_frames = save_frames
        self.save_video = save_video
        self.draw_boxes = draw_boxes
        self.draw_ids = draw_ids
        self.draw_confidence = draw_confidence
        self.fps = fps
        self.realtime_display = realtime_display

        self.frame_info = {}
        self.video_generator = VideoGenerator(fps=fps)

        # 实时显示器
        self.display: Optional[RealtimeDisplay] = None
        if realtime_display:
            self.display = RealtimeDisplay(
                display_fps=fps,
                window_width=window_width,
                window_height=window_height
            )
            self.display.start()

    def process_frame(self, frame_data: FrameData):
        """处理一帧数据，可视化检测结果"""
        try:
            output_frame = frame_data.frame.copy()

            if frame_data.detections and self.draw_boxes:
                output_frame = self._draw_detections(output_frame, frame_data.detections)

            frame_info_text = f"{frame_data.video_id} Frame: {frame_data.frame_id}"
            cv2.putText(output_frame, frame_info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 实时显示
            if self.display and self.realtime_display:
                self.display.update_frame(frame_data.video_id, output_frame)

            if self.save_frames:
                self._save_frame(frame_data, output_frame)

            if self.save_video:
                self._record_frame_info(frame_data, output_frame)

        except Exception as e:
            logger.error(f"Error processing frame: {e}")

    def stop_display(self):
        """停止实时显示"""
        if self.display:
            self.display.stop()

    def is_display_active(self) -> bool:
        """检查显示窗口是否仍然活跃"""
        if self.display:
            return not self.display.stop_event.is_set()
        return True

    def is_user_quit(self) -> bool:
        """检查用户是否主动关闭了窗口"""
        if self.display:
            return self.display.user_quit.is_set()
        return False

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


def create_output_handler(config, realtime_display: bool = False,
                          window_width: int = 1280,
                          window_height: int = 720) -> PipelineOutputHandler:
    """根据配置创建输出处理器"""
    return PipelineOutputHandler(
        output_dir=config.output_dir,
        save_frames=config.save_frames,
        save_video=config.save_video,
        draw_boxes=config.draw_boxes,
        draw_ids=config.draw_ids,
        draw_confidence=config.draw_confidence,
        fps=config.save_fps,
        realtime_display=realtime_display,
        window_width=window_width,
        window_height=window_height,
    )
