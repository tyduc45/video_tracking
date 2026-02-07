"""
视频生成和可视化模块
"""

import cv2
import os
import logging
import threading
import time
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass, field
from queue import Queue, Empty
import numpy as np

from pipeline_data import FrameData
from performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)


@dataclass
class TrackPoint:
    """轨迹点，包含坐标和时间戳"""
    x: float
    y: float
    timestamp: float


@dataclass
class TrackRecord:
    """单个 track_id 的轨迹记录"""
    points: List[TrackPoint] = field(default_factory=list)
    last_update_time: float = 0.0


class TrajectoryManager:
    """
    自定义轨迹管理器

    对于每个视频维护一个字典 dict[int, TrackRecord]：
    - int: track_id
    - TrackRecord: 包含坐标列表和最后更新时间

    逻辑：
    1. 每帧遍历所有 detection，获取 track_id 和坐标
    2. 新 id：创建新条目，记录坐标和时间
    3. 老 id：计算与上次记录的时间差
       - 时间差 > gap_timeout：丢弃历史，重新开始
       - 时间差 <= gap_timeout：添加坐标
    4. 清理逻辑：
       - 每个点根据自身时间戳超时后被删除
       - 整个 track 长时间无更新后被删除
    """

    def __init__(self,
                 gap_timeout: float = 3.0,
                 expire_timeout: float = 3.0,
                 point_expire_timeout: float = 10.0,
                 color_palette=None):
        """
        Args:
            gap_timeout: 同一ID两次记录间隔超过此值（秒）则丢弃历史重新开始
            expire_timeout: ID长时间无更新超过此值（秒）则丢弃整个轨迹
            point_expire_timeout: 单个轨迹点超过此值（秒）后被删除
            color_palette: supervision.ColorPalette 实例，用于保持颜色一致
        """
        self.gap_timeout = gap_timeout
        self.expire_timeout = expire_timeout
        self.point_expire_timeout = point_expire_timeout
        self.color_palette = color_palette

        # dict[track_id, TrackRecord]
        self.tracks: Dict[int, TrackRecord] = {}

    def _get_color(self, track_id: int) -> Tuple[int, int, int]:
        """根据 track_id 获取颜色，与 supervision 检测框颜色一致"""
        if self.color_palette is not None:
            color = self.color_palette.by_idx(track_id)
            # supervision Color 对象的 as_bgr() 返回 BGR 元组
            return color.as_bgr()
        # 备用颜色（不应该走到这里）
        fallback_colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        ]
        return fallback_colors[track_id % len(fallback_colors)]

    def update(self, detections: list, current_time: Optional[float] = None):
        """
        更新轨迹

        Args:
            detections: 检测结果列表，每个元素是 dict，包含 'track_id', 'bbox' 等
            current_time: 当前时间戳，如果为 None 则使用 time.time()
        """
        if current_time is None:
            current_time = time.time()

        # 记录本帧出现的所有 track_id
        current_ids = set()

        for det in detections:
            if not isinstance(det, dict):
                continue

            track_id = det.get('track_id')
            if track_id is None or track_id < 0:
                continue

            bbox = det.get('bbox', [])
            if len(bbox) < 4:
                continue

            current_ids.add(track_id)

            # 计算目标底部中心点作为轨迹点
            x1, y1, x2, y2 = bbox[:4]
            center_x = (x1 + x2) / 2
            bottom_y = y2  # 使用底部中心

            point = TrackPoint(x=center_x, y=bottom_y, timestamp=current_time)

            if track_id not in self.tracks:
                # 新 ID，创建新记录
                self.tracks[track_id] = TrackRecord(
                    points=[point],
                    last_update_time=current_time
                )
            else:
                record = self.tracks[track_id]
                time_gap = current_time - record.last_update_time

                if time_gap > self.gap_timeout:
                    # 时间间隔过大，丢弃历史，重新开始
                    record.points = [point]
                    record.last_update_time = current_time
                else:
                    # 正常添加，点留存在屏幕上直到超时
                    record.points.append(point)
                    record.last_update_time = current_time

        # 清理过期的轨迹（长时间没有更新的 ID）
        self._cleanup_expired(current_time)

    def _cleanup_expired(self, current_time: float):
        """清理过期的轨迹和点"""
        expired_ids = []

        for track_id, record in self.tracks.items():
            # 清理单个超时的点（保留未超时的点）
            record.points = [
                p for p in record.points
                if current_time - p.timestamp <= self.point_expire_timeout
            ]

            # 如果整个 track 长时间无更新，或者所有点都被清理了，删除整个 track
            if (current_time - record.last_update_time > self.expire_timeout
                    or len(record.points) == 0):
                expired_ids.append(track_id)

        for track_id in expired_ids:
            del self.tracks[track_id]

    def draw(self, frame: np.ndarray, thickness: int = 2) -> np.ndarray:
        """
        在帧上绘制所有轨迹

        Args:
            frame: 输入帧
            thickness: 线条粗细

        Returns:
            绘制了轨迹的帧
        """
        result = frame.copy()

        for track_id, record in self.tracks.items():
            if len(record.points) < 2:
                continue

            color = self._get_color(track_id)

            # 连接所有相邻的点
            for i in range(1, len(record.points)):
                pt1 = (int(record.points[i-1].x), int(record.points[i-1].y))
                pt2 = (int(record.points[i].x), int(record.points[i].y))
                cv2.line(result, pt1, pt2, color, thickness)

        return result

    def clear(self):
        """清空所有轨迹"""
        self.tracks.clear()


class PolygonZone:
    """单个视频的多边形区域及计数器"""

    def __init__(self, points: List[Tuple[int, int]]):
        self.points = points          # 多边形顶点列表 (视频本地坐标)
        self.inside_ids: set = set()  # 当前在区域内的 track_id 集合
        self.count: int = 0           # 当前区域内目标数

    def point_in_polygon(self, x: float, y: float) -> bool:
        """射线法判断点是否在多边形内"""
        n = len(self.points)
        if n < 3:
            return False
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = self.points[i]
            xj, yj = self.points[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def update(self, detections: list):
        """每帧更新: 遍历检测框, 计算中心点, 判断进出"""
        current_inside = set()
        for det in detections:
            if not isinstance(det, dict):
                continue
            track_id = det.get('track_id')
            if track_id is None or track_id < 0:
                continue
            bbox = det.get('bbox', [])
            if len(bbox) < 4:
                continue
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            if self.point_in_polygon(cx, cy):
                current_inside.add(track_id)
        self.count = len(current_inside)
        self.inside_ids = current_inside

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """在帧上绘制多边形轮廓 + 计数文字"""
        if len(self.points) < 3:
            return frame
        result = frame.copy()
        pts = np.array(self.points, dtype=np.int32).reshape((-1, 1, 2))
        # 半透明填充
        overlay = result.copy()
        cv2.fillPoly(overlay, [pts], (0, 255, 0))
        cv2.addWeighted(overlay, 0.2, result, 0.8, 0, result)
        # 轮廓线
        cv2.polylines(result, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        # 计数文字 - 放在多边形顶部
        min_x = min(p[0] for p in self.points)
        min_y = min(p[1] for p in self.points)
        text = f"Count: {self.count}"
        cv2.putText(result, text, (min_x, max(min_y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return result


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

        # 布局追踪: { video_id: { 'x': int, 'y': int, 'w': int, 'h': int, 'scale': float } }
        self.video_layout: Dict[str, dict] = {}

        # 多边形绘制状态
        self.drawing = False                    # 是否正在绘制
        self.drawing_points: List[Tuple[int, int]] = []  # 当前绘制中的点 (画布坐标)
        self.drawing_video_id: Optional[str] = None      # 当前绘制所属视频
        self.polygon_zones: Dict[str, PolygonZone] = {}  # video_id -> PolygonZone
        self.polygon_lock = threading.Lock()

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
        self.is_running = False
        logger.info("RealtimeDisplay stopped")

    def update_frame(self, video_id: str, frame: np.ndarray):
        """更新某个视频的最新帧"""
        with self.lock:
            self.current_frames[video_id] = frame.copy()

    def _display_loop(self):
        """显示循环"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

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

                # 绘制多边形覆盖层
                combined = self._draw_polygon_overlay(combined)

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

        # 只销毁自己的窗口
        try:
            cv2.destroyWindow(self.window_name)
        except cv2.error:
            pass

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
        video_ids = [item[0] for item in sorted_items]
        frame_list = [item[1] for item in sorted_items]
        n = len(frame_list)

        if n == 1:
            return self._center_frame_on_canvas(frame_list[0], video_ids[0])

        if n == 2:
            # 两个视频：水平并排，保留原始分辨率，垂直居中对齐
            return self._combine_two_frames(frame_list[0], frame_list[1],
                                            video_ids[0], video_ids[1])

        # 多于2个视频：使用网格布局
        return self._combine_grid_frames(frame_list, video_ids)

    def _combine_two_frames(self, frame1: np.ndarray, frame2: np.ndarray,
                            vid1: str = "", vid2: str = "") -> np.ndarray:
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
        max_width = max(w1, w2)
        total_width = 2 * max_width

        # 创建灰色画布
        canvas = np.full((max_height, total_width, 3), 128, dtype=np.uint8)

        # 放置第一个帧（垂直水平居中）
        y1_offset = (max_height - h1) // 2
        x1_offset = (max_width - w1) // 2
        canvas[y1_offset:y1_offset + h1, x1_offset:x1_offset + w1] = frame1

        # 放置第二个帧（垂直水平居中）
        y2_offset = (max_height - h2) // 2
        x2_offset = (max_width - w2) // 2
        canvas[y2_offset:y2_offset + h2, max_width + x2_offset:max_width + x2_offset + w2] = frame2

        # 记录布局信息
        if vid1:
            self.video_layout[vid1] = {'x': x1_offset, 'y': y1_offset, 'w': w1, 'h': h1, 'scale': 1.0}
        if vid2:
            self.video_layout[vid2] = {'x': max_width + x2_offset, 'y': y2_offset, 'w': w2, 'h': h2, 'scale': 1.0}

        return canvas

    def _combine_grid_frames(self, frame_list: list, video_ids: list = None) -> np.ndarray:
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

            # 记录布局信息
            if video_ids and i < len(video_ids):
                self.video_layout[video_ids[i]] = {
                    'x': x_offset, 'y': y_offset, 'w': w, 'h': h, 'scale': 1.0
                }

        return canvas

    def _center_frame_on_canvas(self, frame: np.ndarray, video_id: str = "") -> np.ndarray:
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

        # 记录布局信息
        if video_id:
            self.video_layout[video_id] = {
                'x': x_offset, 'y': y_offset, 'w': new_w, 'h': new_h, 'scale': scale
            }

        return canvas

    def _canvas_to_video(self, canvas_x: int, canvas_y: int) -> Optional[Tuple[str, int, int]]:
        """画布坐标 -> (video_id, local_x, local_y)，不在任何视频区域则返回 None"""
        for video_id, layout in self.video_layout.items():
            lx = layout['x']
            ly = layout['y']
            lw = layout['w']
            lh = layout['h']
            scale = layout['scale']
            if lx <= canvas_x < lx + lw and ly <= canvas_y < ly + lh:
                # 转换为视频本地坐标（考虑缩放）
                local_x = int((canvas_x - lx) / scale)
                local_y = int((canvas_y - ly) / scale)
                return video_id, local_x, local_y
        return None

    def _video_to_canvas(self, video_id: str, local_x: int, local_y: int) -> Optional[Tuple[int, int]]:
        """视频本地坐标 -> 画布坐标，视频不在布局中则返回 None"""
        layout = self.video_layout.get(video_id)
        if layout is None:
            return None
        scale = layout['scale']
        canvas_x = int(local_x * scale) + layout['x']
        canvas_y = int(local_y * scale) + layout['y']
        return canvas_x, canvas_y

    def _mouse_callback(self, event, x, y, flags, param):
        """鼠标事件回调"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # 左键按下：开始绘制
            result = self._canvas_to_video(x, y)
            if result is not None:
                self.drawing = True
                self.drawing_video_id = result[0]
                self.drawing_points = [(x, y)]

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            # 拖拽中：追加点
            self.drawing_points.append((x, y))

        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            # 左键松开：完成多边形
            self.drawing = False
            if len(self.drawing_points) >= 3 and self.drawing_video_id is not None:
                # 将画布坐标转换为视频本地坐标
                local_points = []
                for cx, cy in self.drawing_points:
                    result = self._canvas_to_video(cx, cy)
                    if result is not None and result[0] == self.drawing_video_id:
                        local_points.append((result[1], result[2]))
                if len(local_points) >= 3:
                    zone = PolygonZone(local_points)
                    with self.polygon_lock:
                        self.polygon_zones[self.drawing_video_id] = zone
                    logger.info(f"Polygon zone created for {self.drawing_video_id} "
                                f"with {len(local_points)} points")
            self.drawing_points = []
            self.drawing_video_id = None

        elif event == cv2.EVENT_RBUTTONDOWN:
            # 右键点击：清除对应视频的多边形
            result = self._canvas_to_video(x, y)
            if result is not None:
                video_id = result[0]
                with self.polygon_lock:
                    if video_id in self.polygon_zones:
                        del self.polygon_zones[video_id]
                        logger.info(f"Polygon zone cleared for {video_id}")

    def _draw_polygon_overlay(self, canvas: np.ndarray) -> np.ndarray:
        """在画布上绘制多边形覆盖层（绘制中的轨迹 + 已完成的多边形）"""
        result = canvas.copy()

        # 绘制正在画的多边形轨迹（画布坐标，无需转换）
        if self.drawing and len(self.drawing_points) >= 2:
            pts = np.array(self.drawing_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(result, [pts], isClosed=False, color=(0, 200, 255), thickness=2)

        # 绘制已完成的多边形（视频本地坐标 -> 画布坐标）
        with self.polygon_lock:
            for video_id, zone in self.polygon_zones.items():
                if len(zone.points) < 3:
                    continue
                canvas_pts = []
                for lx, ly in zone.points:
                    cp = self._video_to_canvas(video_id, lx, ly)
                    if cp is not None:
                        canvas_pts.append(cp)
                if len(canvas_pts) >= 3:
                    pts_arr = np.array(canvas_pts, dtype=np.int32).reshape((-1, 1, 2))
                    # 半透明填充
                    overlay = result.copy()
                    cv2.fillPoly(overlay, [pts_arr], (0, 255, 0))
                    cv2.addWeighted(overlay, 0.15, result, 0.85, 0, result)
                    # 轮廓线
                    cv2.polylines(result, [pts_arr], isClosed=True,
                                  color=(0, 255, 0), thickness=2)
                    # 计数文字
                    min_x = min(p[0] for p in canvas_pts)
                    min_y = min(p[1] for p in canvas_pts)
                    text = f"Count: {zone.count}"
                    cv2.putText(result, text, (min_x, max(min_y - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return result


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
    """Pipeline输出处理器 - 使用 supervision 注解器进行可视化"""

    def __init__(self, output_dir: str = "result",
                 save_frames: bool = True,
                 save_video: bool = True,
                 draw_boxes: bool = True,
                 draw_ids: bool = True,
                 draw_confidence: bool = True,
                 draw_trajectory: bool = True,
                 trajectory_length: int = 30,
                 trajectory_gap_timeout: float = 3.0,
                 trajectory_expire_timeout: float = 3.0,
                 trajectory_point_expire_timeout: float = 10.0,
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
        self.draw_trajectory = draw_trajectory
        self.trajectory_length = trajectory_length
        self.trajectory_gap_timeout = trajectory_gap_timeout
        self.trajectory_expire_timeout = trajectory_expire_timeout
        self.trajectory_point_expire_timeout = trajectory_point_expire_timeout
        self.fps = fps
        self.realtime_display = realtime_display

        self.frame_info = {}
        self.video_generator = VideoGenerator(fps=fps)

        # 初始化 supervision 注解器
        self._init_annotators()

        # 自定义轨迹管理器，每个视频一个
        # {video_id: TrajectoryManager}
        self.trajectory_managers: Dict[str, TrajectoryManager] = {}

        # 实时显示器
        self.display: Optional[RealtimeDisplay] = None
        if realtime_display:
            self.display = RealtimeDisplay(
                display_fps=fps,
                window_width=window_width,
                window_height=window_height
            )
            self.display.start()

    def _init_annotators(self):
        """初始化 supervision 注解器"""
        import supervision as sv
        self.sv = sv

        # 颜色按 track_id 分配
        self.color_lookup = sv.ColorLookup.TRACK

        # 检测框注解器（无状态，可共享）
        self.box_annotator = sv.RoundBoxAnnotator(
            color=sv.ColorPalette.DEFAULT,
            thickness=2,
            color_lookup=self.color_lookup
        )

        # 标签注解器（无状态，可共享）
        self.label_annotator = sv.LabelAnnotator(
            color=sv.ColorPalette.DEFAULT,
            text_color=sv.Color.BLACK,
            text_scale=0.5,
            text_thickness=1,
            text_padding=5,
            color_lookup=self.color_lookup
        )

        logger.info("Supervision annotators initialized")

    def _get_trajectory_manager(self, video_id: str) -> TrajectoryManager:
        """获取指定视频的轨迹管理器（懒加载）"""
        if video_id not in self.trajectory_managers:
            self.trajectory_managers[video_id] = TrajectoryManager(
                gap_timeout=self.trajectory_gap_timeout,
                expire_timeout=self.trajectory_expire_timeout,
                point_expire_timeout=self.trajectory_point_expire_timeout,
                color_palette=self.sv.ColorPalette.DEFAULT
            )
        return self.trajectory_managers[video_id]

    def _get_polygon_zone(self, video_id: str) -> Optional[PolygonZone]:
        """线程安全地获取指定视频的多边形区域"""
        if self.display is None:
            return None
        with self.display.polygon_lock:
            return self.display.polygon_zones.get(video_id)

    def process_frame(self, frame_data: FrameData):
        """处理一帧数据，可视化检测结果"""
        try:
            output_frame = frame_data.frame.copy()

            if frame_data.detections and self.draw_boxes:
                output_frame = self._draw_detections(
                    output_frame, frame_data.detections, frame_data.video_id
                )

            # 多边形区域检测计数
            polygon_zone = self._get_polygon_zone(frame_data.video_id)
            if polygon_zone is not None:
                polygon_zone.update(frame_data.detections or [])
                output_frame = polygon_zone.draw(output_frame)

            frame_info_text = f"{frame_data.video_id} Frame: {frame_data.frame_id}"
            cv2.putText(output_frame, frame_info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 实时显示
            if self.display and self.realtime_display:
                self.display.update_frame(frame_data.video_id, output_frame)

            # 性能探针: 帧处理完成
            PerformanceMonitor.probe(frame_data.video_id, frame_data.frame_id, "end")

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

    def _detections_to_sv(self, detections: list):
        """将检测结果列表转换为 supervision.Detections 对象"""
        if not detections:
            return self.sv.Detections.empty()

        xyxy = []
        confidence = []
        class_id = []
        tracker_id = []
        class_names = []

        for det in detections:
            if not isinstance(det, dict):
                continue

            bbox = det.get('bbox', [])
            if len(bbox) < 4:
                continue

            xyxy.append(bbox[:4])
            confidence.append(det.get('confidence', 0.0))
            class_id.append(det.get('class_id', 0))
            class_names.append(det.get('class_name', 'unknown'))

            tid = det.get('track_id')
            tracker_id.append(tid if tid is not None else -1)

        if not xyxy:
            return self.sv.Detections.empty()

        # 构建 Detections 对象
        return self.sv.Detections(
            xyxy=np.array(xyxy, dtype=np.float32),
            confidence=np.array(confidence, dtype=np.float32),
            class_id=np.array(class_id, dtype=int),
            tracker_id=np.array(tracker_id, dtype=int),
            data={'class_name': np.array(class_names)}
        )

    def _build_labels(self, detections) -> list:
        """构建标签列表"""
        labels = []
        class_names = detections.data.get('class_name', [])

        for i in range(len(detections.xyxy)):
            class_name = class_names[i] if i < len(class_names) else 'unknown'
            label = f"{class_name}"

            if self.draw_confidence and detections.confidence is not None:
                label += f" {detections.confidence[i]:.2f}"

            if self.draw_ids and detections.tracker_id is not None:
                tid = detections.tracker_id[i]
                if tid >= 0:
                    label += f" #{tid}"

            labels.append(label)

        return labels

    def _draw_detections(self, frame: np.ndarray, detections, video_id: str = "default") -> np.ndarray:
        """使用 supervision 注解器绘制检测结果和自定义轨迹"""
        # 如果是 ultralytics 的原生结果，直接使用其 plot 方法
        if hasattr(detections, 'plot'):
            return detections.plot()

        if not isinstance(detections, list) or not detections:
            return frame

        # 转换为 supervision.Detections
        sv_detections = self._detections_to_sv(detections)

        if len(sv_detections.xyxy) == 0:
            return frame

        annotated_frame = frame.copy()

        # 使用自定义轨迹管理器绘制轨迹（需要在检测框之前绘制，这样轨迹在框下面）
        if self.draw_trajectory:
            trajectory_manager = self._get_trajectory_manager(video_id)
            # 更新轨迹（传入原始 detections 列表）
            trajectory_manager.update(detections)
            # 绘制轨迹
            annotated_frame = trajectory_manager.draw(annotated_frame, thickness=2)

        # 绘制检测框
        if self.draw_boxes:
            annotated_frame = self.box_annotator.annotate(
                scene=annotated_frame,
                detections=sv_detections
            )

        # 绘制标签
        if self.draw_ids or self.draw_confidence:
            labels = self._build_labels(sv_detections)
            annotated_frame = self.label_annotator.annotate(
                scene=annotated_frame,
                detections=sv_detections,
                labels=labels
            )

        return annotated_frame

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
        draw_trajectory=getattr(config, 'draw_trajectory', True),
        trajectory_length=getattr(config, 'trajectory_length', 30),
        trajectory_gap_timeout=getattr(config, 'trajectory_gap_timeout', 3.0),
        trajectory_expire_timeout=getattr(config, 'trajectory_expire_timeout', 3.0),
        trajectory_point_expire_timeout=getattr(config, 'trajectory_point_expire_timeout', 10.0),
        fps=config.save_fps,
        realtime_display=realtime_display,
        window_width=window_width,
        window_height=window_height,
    )
