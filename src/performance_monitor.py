"""
性能监控模块 - 探针模式实现

采用探针函数在代码关键位置插入计时点，自动收集数据并实时绘制折线图。
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)


@dataclass
class VideoTimer:
    """每个视频流的独立计时器"""
    video_id: str
    frame_id: int = 0                    # 当前帧ID
    start_time: float = 0.0              # 开始时间戳(高精度)
    latencies: List[float] = field(default_factory=list)  # 历史延迟记录
    is_active: bool = True               # 是否活跃
    lock: threading.Lock = field(default_factory=threading.Lock)
    total_frames: int = 0                # 总帧数
    total_time: float = 0.0              # 总处理时间
    first_frame_time: float = 0.0        # 第一帧时间(用于计算整体FPS)
    last_frame_time: float = 0.0         # 最后一帧时间


class PerformanceWindow:
    """独立的性能监控窗口"""

    def __init__(self, window_name: str = "Performance Monitor",
                 num_videos: int = 1,
                 chart_width: int = 300,
                 chart_height: int = 150):
        self.window_name = window_name
        self.num_videos = num_videos
        self.chart_width = chart_width
        self.chart_height = chart_height

        # 计算网格布局
        self.cols = int(np.ceil(np.sqrt(num_videos)))
        self.rows = int(np.ceil(num_videos / self.cols))

        # 每个子图区域的大小（包含图表和文字）
        self.cell_width = chart_width + 40   # 左右边距
        self.cell_height = chart_height + 80  # 上下边距 + 文字空间

        # 画布大小
        self.canvas_width = self.cell_width * self.cols
        self.canvas_height = self.cell_height * self.rows

        # 图表数据: video_id -> deque of latency values (ms)
        self.chart_data: Dict[str, deque] = {}

        # 最终统计: video_id -> avg_latency_ms
        self.final_stats: Dict[str, float] = {}

        # 当前延迟: video_id -> latency (seconds)
        self.current_latency: Dict[str, float] = {}

        # video_id 到 index 的映射
        self.video_index: Dict[str, int] = {}

        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.display_thread: Optional[threading.Thread] = None
        self.is_running = False

    def register_video(self, video_id: str):
        """注册一个视频"""
        with self.lock:
            if video_id not in self.video_index:
                self.video_index[video_id] = len(self.video_index)
                self.chart_data[video_id] = deque(maxlen=100)

    def start(self):
        """启动显示线程"""
        if self.is_running:
            return

        self.is_running = True
        self.stop_event.clear()
        self.display_thread = threading.Thread(
            target=self._display_loop,
            name="PerformanceWindow",
            daemon=True
        )
        self.display_thread.start()
        logger.info(f"PerformanceWindow started: {self.window_name}")

    def stop(self):
        """停止显示"""
        logger.info("PerformanceWindow stopping...")
        self.stop_event.set()
        if self.display_thread and self.display_thread.is_alive():
            self.display_thread.join(timeout=3.0)
            if self.display_thread.is_alive():
                logger.warning("PerformanceWindow display thread did not exit in time")
        self.is_running = False
        logger.info("PerformanceWindow stopped")

    def update(self, video_id: str, latency: float):
        """更新视频的延迟数据"""
        with self.lock:
            if video_id not in self.chart_data:
                self.register_video(video_id)
            latency_ms = latency * 1000  # 转换为毫秒
            self.chart_data[video_id].append(latency_ms)
            self.current_latency[video_id] = latency

    def set_final_stats(self, video_id: str, avg_latency_ms: float):
        """设置视频的最终统计（平均延迟，毫秒）"""
        with self.lock:
            self.final_stats[video_id] = avg_latency_ms

    def _display_loop(self):
        """显示循环"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.canvas_width, self.canvas_height)

        while not self.stop_event.is_set():
            canvas = self._render()
            cv2.imshow(self.window_name, canvas)

            # 使用较短的 waitKey 超时，以便快速响应 stop_event
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q') or key == 27:
                break

            # 检查窗口是否被关闭
            try:
                window_visible = cv2.getWindowProperty(
                    self.window_name, cv2.WND_PROP_VISIBLE
                )
                if window_visible < 1:
                    break
            except cv2.error:
                break

        # 在创建窗口的同一线程中销毁窗口
        logger.info(f"PerformanceWindow display loop exiting, destroying window: {self.window_name}")
        try:
            cv2.destroyWindow(self.window_name)
        except cv2.error as e:
            logger.warning(f"Failed to destroy window: {e}")

    def _render(self) -> np.ndarray:
        """渲染整个监控画面"""
        # 创建深灰色背景
        canvas = np.full((self.canvas_height, self.canvas_width, 3), 40, dtype=np.uint8)

        with self.lock:
            for video_id, index in self.video_index.items():
                row = index // self.cols
                col = index % self.cols

                # 计算该视频区域的位置
                cell_x = col * self.cell_width
                cell_y = row * self.cell_height

                # 绘制该视频的监控面板
                self._draw_video_panel(canvas, video_id, cell_x, cell_y)

        return canvas

    def _draw_video_panel(self, canvas: np.ndarray, video_id: str,
                          cell_x: int, cell_y: int):
        """绘制单个视频的监控面板"""
        margin = 20
        chart_x = cell_x + margin
        chart_y = cell_y + margin + 20  # 为标题留空间

        # 绘制视频标题
        cv2.putText(canvas, video_id, (chart_x, cell_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 绘制图表背景
        cv2.rectangle(canvas,
                      (chart_x, chart_y),
                      (chart_x + self.chart_width, chart_y + self.chart_height),
                      (60, 60, 60), -1)

        # 绘制图表边框
        cv2.rectangle(canvas,
                      (chart_x, chart_y),
                      (chart_x + self.chart_width, chart_y + self.chart_height),
                      (100, 100, 100), 1)

        # 获取数据 (latency in ms)
        data = self.chart_data.get(video_id, deque())
        latency = self.current_latency.get(video_id, 0.0)
        final_latency = self.final_stats.get(video_id)

        # 绘制延迟折线图
        if len(data) >= 2:
            self._draw_line_chart(canvas, list(data),
                                  chart_x, chart_y,
                                  self.chart_width, self.chart_height)

        # 绘制当前延迟值（图表内左上角）
        latency_ms = latency * 1000
        latency_text = f"Latency: {latency_ms:.1f}ms"
        cv2.putText(canvas, latency_text,
                    (chart_x + 5, chart_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # 绘制统计信息（图表下方）
        info_y = chart_y + self.chart_height + 20

        if final_latency is not None:
            avg_text = f"Avg Latency: {final_latency:.2f}ms"
            cv2.putText(canvas, avg_text, (chart_x, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        elif data:
            avg_latency = sum(data) / len(data)
            avg_text = f"Avg Latency: {avg_latency:.2f}ms"
            cv2.putText(canvas, avg_text, (chart_x, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    def _draw_line_chart(self, canvas: np.ndarray, data_points: List[float],
                         x: int, y: int, w: int, h: int):
        """在指定区域绘制折线图"""
        if len(data_points) < 2:
            return

        max_val = max(data_points)
        min_val = min(data_points)
        range_val = max_val - min_val if max_val > min_val else 1

        # 添加一些边距
        padding = 5
        chart_h = h - 2 * padding
        chart_w = w - 2 * padding

        points = []
        for i, val in enumerate(data_points):
            px = x + padding + int(i * chart_w / (len(data_points) - 1))
            py = y + padding + chart_h - int((val - min_val) * chart_h / range_val)
            points.append((px, py))

        # 绘制折线
        for i in range(len(points) - 1):
            cv2.line(canvas, points[i], points[i + 1], (0, 255, 0), 2)

        # 绘制数据点
        for pt in points[-10:]:  # 只显示最近10个点
            cv2.circle(canvas, pt, 2, (0, 200, 0), -1)


class PerformanceMonitor:
    """全局性能监控器 - 单例模式"""
    _instance: Optional['PerformanceMonitor'] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, num_videos: int = 1, enabled: bool = True):
        if self._initialized:
            return

        self.enabled = enabled
        self.timers: Dict[str, VideoTimer] = {}
        self.timers_lock = threading.Lock()

        self.window: Optional[PerformanceWindow] = None
        if enabled:
            self.window = PerformanceWindow(num_videos=num_videos)

        self._initialized = True
        logger.info(f"PerformanceMonitor initialized, enabled={enabled}")

    @classmethod
    def get_instance(cls) -> Optional['PerformanceMonitor']:
        """获取单例实例"""
        return cls._instance

    @classmethod
    def reset(cls):
        """重置单例（用于测试）"""
        with cls._lock:
            if cls._instance is not None:
                if cls._instance.window:
                    cls._instance.window.stop()
                cls._instance = None

    def start(self):
        """启动性能监控窗口"""
        if self.window and self.enabled:
            self.window.start()

    def stop(self):
        """停止性能监控"""
        if self.window:
            self.window.stop()

    def _get_or_create_timer(self, video_id: str) -> VideoTimer:
        """获取或创建视频计时器"""
        with self.timers_lock:
            if video_id not in self.timers:
                self.timers[video_id] = VideoTimer(video_id=video_id)
                if self.window:
                    self.window.register_video(video_id)
            return self.timers[video_id]

    @staticmethod
    def probe(video_id: str, frame_id: int, action: str):
        """
        探针函数 - 唯一的外部接口

        Args:
            video_id: 视频标识
            frame_id: 帧序号
            action: "start" | "end" | "finish"
                - start: 帧开始处理（Reader读取后）
                - end: 帧处理完成（显示后）
                - finish: 视频结束，计算最终统计
        """
        instance = PerformanceMonitor.get_instance()
        if instance is None or not instance.enabled:
            return

        if action == "start":
            instance._handle_start(video_id, frame_id)
        elif action == "end":
            instance._handle_end(video_id, frame_id)
        elif action == "finish":
            instance._handle_finish(video_id)

    def _handle_start(self, video_id: str, frame_id: int):
        """处理帧开始事件"""
        timer = self._get_or_create_timer(video_id)

        with timer.lock:
            timer.frame_id = frame_id
            timer.start_time = time.perf_counter()

            if timer.total_frames == 0:
                timer.first_frame_time = timer.start_time

    def _handle_end(self, video_id: str, frame_id: int):
        """处理帧结束事件"""
        end_time = time.perf_counter()

        timer = self._get_or_create_timer(video_id)

        with timer.lock:
            if timer.start_time == 0:
                return  # 没有对应的start

            # 计算延迟
            latency = end_time - timer.start_time

            # 更新统计
            timer.latencies.append(latency)
            timer.total_frames += 1
            timer.total_time += latency
            timer.last_frame_time = end_time

            # 更新窗口（传入延迟值）
            if self.window:
                self.window.update(video_id, latency)

            # 重置start_time
            timer.start_time = 0

    def _handle_finish(self, video_id: str):
        """处理视频结束事件"""
        timer = self.timers.get(video_id)
        if timer is None:
            return

        with timer.lock:
            timer.is_active = False

            # 计算最终统计
            if timer.total_frames > 0 and timer.last_frame_time > timer.first_frame_time:
                # 计算平均延迟（毫秒）
                avg_latency_ms = (timer.total_time / timer.total_frames) * 1000
                # 计算平均FPS
                elapsed = timer.last_frame_time - timer.first_frame_time
                avg_fps = timer.total_frames / elapsed if elapsed > 0 else 0

                if self.window:
                    self.window.set_final_stats(video_id, avg_latency_ms)

                logger.info(f"[{video_id}] Performance stats: "
                            f"frames={timer.total_frames}, "
                            f"avg_fps={avg_fps:.2f}, "
                            f"avg_latency={avg_latency_ms:.2f}ms")

    def get_stats(self, video_id: str) -> Optional[dict]:
        """获取视频的性能统计"""
        timer = self.timers.get(video_id)
        if timer is None:
            return None

        with timer.lock:
            avg_latency = timer.total_time / timer.total_frames if timer.total_frames > 0 else 0
            elapsed = timer.last_frame_time - timer.first_frame_time if timer.first_frame_time > 0 else 0
            avg_fps = timer.total_frames / elapsed if elapsed > 0 else 0

            return {
                'video_id': video_id,
                'total_frames': timer.total_frames,
                'total_time': timer.total_time,
                'avg_latency': avg_latency,
                'avg_fps': avg_fps,
                'is_active': timer.is_active,
            }

    def get_all_stats(self) -> Dict[str, dict]:
        """获取所有视频的性能统计"""
        stats = {}
        for video_id in self.timers:
            stat = self.get_stats(video_id)
            if stat:
                stats[video_id] = stat
        return stats
