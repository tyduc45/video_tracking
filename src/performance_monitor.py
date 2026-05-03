"""
性能监控模块 - 探针模式实现

采集三类指标：
- processing_fps: 帧从读入到处理完成的吞吐率
- display_fps: 显示线程实际刷新新帧的频率
- e2e_latency: 按 frame_id 计算读入到真实展示或保存完成的端到端延迟
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - depends on local GUI/runtime environment
    cv2 = None

logger = logging.getLogger(__name__)


HISTORY_SIZE = 120
PENDING_FRAME_LIMIT = 3000


def _rolling_fps(timestamps: deque) -> float:
    """根据滚动时间戳计算 FPS。"""
    if len(timestamps) < 2:
        return 0.0
    elapsed = timestamps[-1] - timestamps[0]
    if elapsed <= 0:
        return 0.0
    return (len(timestamps) - 1) / elapsed


@dataclass
class VideoTimer:
    """每个视频流的独立计时器。"""

    video_id: str
    lock: threading.Lock = field(default_factory=threading.Lock)
    is_active: bool = True

    # frame_id -> read timestamp. 不能只存一个 start_time，否则批处理/多线程会互相覆盖。
    start_times: Dict[int, float] = field(default_factory=dict)
    e2e_recorded_frames: Set[int] = field(default_factory=set)
    display_recorded_frames: Set[int] = field(default_factory=set)

    process_timestamps: deque = field(default_factory=lambda: deque(maxlen=HISTORY_SIZE))
    display_timestamps: deque = field(default_factory=lambda: deque(maxlen=HISTORY_SIZE))

    processing_fps_history: deque = field(default_factory=lambda: deque(maxlen=HISTORY_SIZE))
    display_fps_history: deque = field(default_factory=lambda: deque(maxlen=HISTORY_SIZE))
    e2e_latency_history_ms: deque = field(default_factory=lambda: deque(maxlen=HISTORY_SIZE))
    processing_latency_history_ms: deque = field(default_factory=lambda: deque(maxlen=HISTORY_SIZE))

    total_started: int = 0
    total_processed: int = 0
    total_displayed: int = 0
    total_e2e_samples: int = 0
    total_e2e_latency: float = 0.0
    total_processing_latency: float = 0.0

    first_start_time: float = 0.0
    last_process_time: float = 0.0
    last_display_time: float = 0.0

    current_processing_fps: float = 0.0
    current_display_fps: float = 0.0
    current_e2e_latency_ms: float = 0.0
    current_processing_latency_ms: float = 0.0


class OpenCVPerformanceWindow:
    """OpenCV 兜底性能窗口。

    当 PySide6/pyqtgraph 未安装时使用，保证旧运行方式仍然可用。
    """

    def __init__(self, snapshot_provider: Callable[[], Dict[str, dict]],
                 window_name: str = "Performance Monitor",
                 num_videos: int = 1):
        self.snapshot_provider = snapshot_provider
        self.window_name = window_name
        self.num_videos = num_videos
        self.chart_width = 300
        self.chart_height = 110
        self.cell_width = self.chart_width + 50
        self.cell_height = self.chart_height * 3 + 120
        self.cols = int(np.ceil(np.sqrt(max(num_videos, 1))))
        self.rows = int(np.ceil(max(num_videos, 1) / self.cols))
        self.canvas_width = self.cell_width * self.cols
        self.canvas_height = self.cell_height * self.rows
        self.stop_event = threading.Event()
        self.display_thread: Optional[threading.Thread] = None
        self.is_running = False

    def register_video(self, video_id: str):
        """兼容 PyQt 窗口接口，OpenCV 版本每帧从 snapshot 重绘。"""

    def start(self):
        if cv2 is None:
            logger.warning("OpenCV is not available; performance window cannot be shown")
            return
        if self.is_running:
            return
        self.is_running = True
        self.stop_event.clear()
        self.display_thread = threading.Thread(
            target=self._display_loop,
            name="OpenCVPerformanceWindow",
            daemon=True,
        )
        self.display_thread.start()
        logger.info("OpenCV performance window started")

    def stop(self):
        self.stop_event.set()
        if self.display_thread and self.display_thread.is_alive():
            self.display_thread.join(timeout=3.0)
        self.is_running = False
        logger.info("OpenCV performance window stopped")

    def _display_loop(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.canvas_width, self.canvas_height)

        while not self.stop_event.is_set():
            canvas = self._render()
            cv2.imshow(self.window_name, canvas)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q") or key == 27:
                break
            try:
                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

        try:
            cv2.destroyWindow(self.window_name)
        except cv2.error:
            pass

    def _render(self) -> np.ndarray:
        snapshot = self.snapshot_provider()
        video_ids = sorted(snapshot.keys())
        if not video_ids:
            video_ids = ["waiting"]

        canvas = np.full((self.canvas_height, self.canvas_width, 3), 38, dtype=np.uint8)
        for index, video_id in enumerate(video_ids):
            row = index // self.cols
            col = index % self.cols
            x = col * self.cell_width + 20
            y = row * self.cell_height + 25
            self._draw_video_panel(canvas, video_id, snapshot.get(video_id, {}), x, y)
        return canvas

    def _draw_video_panel(self, canvas: np.ndarray, video_id: str, data: dict, x: int, y: int):
        cv2.putText(canvas, video_id, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1)
        charts = [
            ("processing_fps", "Processing FPS", (80, 220, 255)),
            ("display_fps", "Display FPS", (120, 255, 120)),
            ("e2e_latency_ms", "E2E Latency ms", (255, 190, 90)),
        ]
        for idx, (key, title, color) in enumerate(charts):
            chart_y = y + 25 + idx * (self.chart_height + 28)
            values = data.get(f"{key}_history", [])
            current = data.get(key, 0.0)
            cv2.rectangle(canvas, (x, chart_y),
                          (x + self.chart_width, chart_y + self.chart_height),
                          (58, 58, 58), -1)
            cv2.rectangle(canvas, (x, chart_y),
                          (x + self.chart_width, chart_y + self.chart_height),
                          (105, 105, 105), 1)
            cv2.putText(canvas, f"{title}: {current:.2f}", (x + 6, chart_y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
            self._draw_line_chart(canvas, values, x, chart_y, self.chart_width,
                                  self.chart_height, color)

    @staticmethod
    def _draw_line_chart(canvas: np.ndarray, values: List[float],
                         x: int, y: int, w: int, h: int, color):
        if len(values) < 2:
            return
        max_val = max(values)
        min_val = min(values)
        value_range = max(max_val - min_val, 1e-6)
        padding = 8
        points = []
        for i, value in enumerate(values):
            px = x + padding + int(i * (w - 2 * padding) / (len(values) - 1))
            py = y + h - padding - int((value - min_val) * (h - 2 * padding) / value_range)
            points.append((px, py))
        for i in range(len(points) - 1):
            cv2.line(canvas, points[i], points[i + 1], color, 2)


class PyQtPerformanceWindow:
    """PySide6 + pyqtgraph 实时性能窗口。"""

    def __init__(self, snapshot_provider: Callable[[], Dict[str, dict]],
                 window_name: str = "Performance Monitor",
                 num_videos: int = 1):
        self.snapshot_provider = snapshot_provider
        self.window_name = window_name
        self.num_videos = num_videos
        self.stop_event = threading.Event()
        self.display_thread: Optional[threading.Thread] = None
        self.is_running = False

    @staticmethod
    def is_available() -> bool:
        try:
            import pyqtgraph  # noqa: F401
            from PySide6 import QtWidgets  # noqa: F401
            return True
        except Exception:
            return False

    def register_video(self, video_id: str):
        """Qt 窗口通过定时 snapshot 自动发现视频源。"""

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self.stop_event.clear()
        self.display_thread = threading.Thread(
            target=self._display_loop,
            name="PyQtPerformanceWindow",
            daemon=True,
        )
        self.display_thread.start()
        logger.info("PyQt performance window started")

    def stop(self):
        self.stop_event.set()
        if self.display_thread and self.display_thread.is_alive():
            self.display_thread.join(timeout=3.0)
        self.is_running = False
        logger.info("PyQt performance window stopped")

    def _display_loop(self):
        from PySide6 import QtCore, QtWidgets
        import pyqtgraph as pg

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        pg.setConfigOptions(antialias=True, background="#202327", foreground="#d9dee7")

        window = QtWidgets.QMainWindow()
        window.setWindowTitle(self.window_name)
        window.resize(1280, 820)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        root = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(root)
        grid.setContentsMargins(12, 12, 12, 12)
        grid.setSpacing(12)
        scroll.setWidget(root)
        window.setCentralWidget(scroll)

        panels: Dict[str, dict] = {}

        def make_panel(video_id: str, index: int):
            panel = QtWidgets.QFrame()
            panel.setFrameShape(QtWidgets.QFrame.StyledPanel)
            panel.setStyleSheet(
                "QFrame { background: #282c31; border: 1px solid #3b4148; border-radius: 6px; }"
                "QLabel { color: #e8edf5; border: 0; }"
            )
            layout = QtWidgets.QVBoxLayout(panel)
            title = QtWidgets.QLabel(video_id)
            title.setStyleSheet("font-size: 15px; font-weight: 600;")
            summary = QtWidgets.QLabel("processing_fps 0.00   display_fps 0.00   e2e_latency 0.00ms")
            summary.setStyleSheet("color: #b8c0cc;")
            layout.addWidget(title)
            layout.addWidget(summary)

            plot_defs = [
                ("processing_fps", "Processing FPS", "#4cc9f0"),
                ("display_fps", "Display FPS", "#80ed99"),
                ("e2e_latency_ms", "E2E Latency (ms)", "#f9c74f"),
            ]
            plots = {}
            for key, name, color in plot_defs:
                plot = pg.PlotWidget(title=name)
                plot.setMinimumHeight(150)
                plot.showGrid(x=True, y=True, alpha=0.25)
                curve = plot.plot([], [], pen=pg.mkPen(color=color, width=2))
                layout.addWidget(plot)
                plots[key] = curve

            cols = int(np.ceil(np.sqrt(max(self.num_videos, len(panels) + 1, 1))))
            row = index // cols
            col = index % cols
            grid.addWidget(panel, row, col)
            panels[video_id] = {"summary": summary, "plots": plots}

        def refresh():
            if self.stop_event.is_set():
                window.close()
                app.quit()
                return

            snapshot = self.snapshot_provider()
            for video_id in sorted(snapshot.keys()):
                if video_id not in panels:
                    make_panel(video_id, len(panels))

                data = snapshot[video_id]
                panels[video_id]["summary"].setText(
                    f"processing_fps {data.get('processing_fps', 0.0):.2f}   "
                    f"display_fps {data.get('display_fps', 0.0):.2f}   "
                    f"e2e_latency {data.get('e2e_latency_ms', 0.0):.2f}ms"
                )
                for key, curve in panels[video_id]["plots"].items():
                    values = data.get(f"{key}_history", [])
                    curve.setData(list(range(len(values))), values)

        timer = QtCore.QTimer()
        timer.timeout.connect(refresh)
        timer.start(250)

        window.show()
        app.exec()
        self.stop_event.set()


class PerformanceMonitor:
    """全局性能监控器 - 单例模式。"""

    _instance: Optional["PerformanceMonitor"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, num_videos: int = 1, enabled: bool = True,
                 window_backend: str = "auto"):
        if self._initialized:
            return

        self.enabled = enabled
        self.timers: Dict[str, VideoTimer] = {}
        self.timers_lock = threading.Lock()
        self.window = None

        if enabled:
            use_pyqt = window_backend == "pyqt" or (
                window_backend == "auto" and PyQtPerformanceWindow.is_available()
            )
            if use_pyqt:
                self.window = PyQtPerformanceWindow(self.get_all_stats, num_videos=num_videos)
            else:
                if window_backend == "pyqt":
                    logger.warning("PySide6/pyqtgraph not available, falling back to OpenCV")
                self.window = OpenCVPerformanceWindow(self.get_all_stats, num_videos=num_videos)

        self._initialized = True
        logger.info(f"PerformanceMonitor initialized, enabled={enabled}")

    @classmethod
    def get_instance(cls) -> Optional["PerformanceMonitor"]:
        return cls._instance

    @classmethod
    def reset(cls):
        with cls._lock:
            if cls._instance is not None:
                if cls._instance.window:
                    cls._instance.window.stop()
                cls._instance = None

    def start(self):
        if self.window and self.enabled:
            self.window.start()

    def stop(self):
        if self.window:
            self.window.stop()

    def _get_or_create_timer(self, video_id: str) -> VideoTimer:
        with self.timers_lock:
            if video_id not in self.timers:
                self.timers[video_id] = VideoTimer(video_id=video_id)
                if self.window:
                    self.window.register_video(video_id)
            return self.timers[video_id]

    @staticmethod
    def probe(video_id: str, frame_id: int, action: str):
        """
        探针函数 - 唯一的外部接口。

        action:
            start/process_start: Reader 读入帧
            end/process_end: 推理、追踪、可视化绘制完成
            display: 显示线程完成一次包含该 frame_id 的 imshow
            save: 帧保存/记录完成，用于无实时显示时的 e2e_latency
            finish: 视频处理结束
        """
        instance = PerformanceMonitor.get_instance()
        if instance is None or not instance.enabled:
            return

        if action in ("start", "process_start"):
            instance._handle_start(video_id, frame_id)
        elif action in ("end", "process_end"):
            instance._handle_process_end(video_id, frame_id)
        elif action == "display":
            instance._handle_display(video_id, frame_id)
        elif action == "save":
            instance._handle_e2e_complete(video_id, frame_id)
        elif action == "finish":
            instance._handle_finish(video_id)

    def _handle_start(self, video_id: str, frame_id: int):
        now = time.perf_counter()
        timer = self._get_or_create_timer(video_id)

        with timer.lock:
            timer.start_times[frame_id] = now
            timer.total_started += 1
            if timer.first_start_time == 0:
                timer.first_start_time = now

            if len(timer.start_times) > PENDING_FRAME_LIMIT:
                oldest_ids = sorted(timer.start_times)[: len(timer.start_times) - PENDING_FRAME_LIMIT]
                for old_id in oldest_ids:
                    timer.start_times.pop(old_id, None)
                    timer.e2e_recorded_frames.discard(old_id)

    def _handle_process_end(self, video_id: str, frame_id: int):
        now = time.perf_counter()
        timer = self._get_or_create_timer(video_id)

        with timer.lock:
            start_time = timer.start_times.get(frame_id)
            if start_time is not None:
                latency = now - start_time
                latency_ms = latency * 1000
                timer.current_processing_latency_ms = latency_ms
                timer.processing_latency_history_ms.append(latency_ms)
                timer.total_processing_latency += latency

            timer.process_timestamps.append(now)
            timer.current_processing_fps = _rolling_fps(timer.process_timestamps)
            timer.processing_fps_history.append(timer.current_processing_fps)
            timer.total_processed += 1
            timer.last_process_time = now

    def _handle_display(self, video_id: str, frame_id: int):
        now = time.perf_counter()
        timer = self._get_or_create_timer(video_id)

        with timer.lock:
            if frame_id in timer.display_recorded_frames:
                return
            timer.display_recorded_frames.add(frame_id)
            timer.display_timestamps.append(now)
            timer.current_display_fps = _rolling_fps(timer.display_timestamps)
            timer.display_fps_history.append(timer.current_display_fps)
            timer.total_displayed += 1
            timer.last_display_time = now
            if len(timer.display_recorded_frames) > PENDING_FRAME_LIMIT:
                timer.display_recorded_frames = set(
                    sorted(timer.display_recorded_frames)[-PENDING_FRAME_LIMIT:]
                )

        self._handle_e2e_complete(video_id, frame_id, now=now)

    def _handle_e2e_complete(self, video_id: str, frame_id: int,
                             now: Optional[float] = None):
        complete_time = now if now is not None else time.perf_counter()
        timer = self._get_or_create_timer(video_id)

        with timer.lock:
            if frame_id in timer.e2e_recorded_frames:
                return
            start_time = timer.start_times.get(frame_id)
            if start_time is None:
                return

            latency = complete_time - start_time
            latency_ms = latency * 1000
            timer.current_e2e_latency_ms = latency_ms
            timer.e2e_latency_history_ms.append(latency_ms)
            timer.total_e2e_latency += latency
            timer.total_e2e_samples += 1
            timer.e2e_recorded_frames.add(frame_id)

            # e2e 已完成后可以释放该帧起点，避免长时间运行内存增长。
            timer.start_times.pop(frame_id, None)
            if len(timer.e2e_recorded_frames) > PENDING_FRAME_LIMIT:
                timer.e2e_recorded_frames = set(sorted(timer.e2e_recorded_frames)[-PENDING_FRAME_LIMIT:])

    def _handle_finish(self, video_id: str):
        timer = self.timers.get(video_id)
        if timer is None:
            return

        with timer.lock:
            timer.is_active = False
            avg_e2e_ms = (
                timer.total_e2e_latency / timer.total_e2e_samples * 1000
                if timer.total_e2e_samples > 0 else 0.0
            )
            avg_processing_ms = (
                timer.total_processing_latency / timer.total_processed * 1000
                if timer.total_processed > 0 else 0.0
            )
            logger.info(
                f"[{video_id}] Performance stats: "
                f"processed={timer.total_processed}, displayed={timer.total_displayed}, "
                f"processing_fps={timer.current_processing_fps:.2f}, "
                f"display_fps={timer.current_display_fps:.2f}, "
                f"avg_processing_latency={avg_processing_ms:.2f}ms, "
                f"avg_e2e_latency={avg_e2e_ms:.2f}ms"
            )

    def get_stats(self, video_id: str) -> Optional[dict]:
        timer = self.timers.get(video_id)
        if timer is None:
            return None

        with timer.lock:
            avg_e2e_latency = (
                timer.total_e2e_latency / timer.total_e2e_samples
                if timer.total_e2e_samples > 0 else 0.0
            )
            avg_processing_latency = (
                timer.total_processing_latency / timer.total_processed
                if timer.total_processed > 0 else 0.0
            )
            return {
                "video_id": video_id,
                "is_active": timer.is_active,
                "total_started": timer.total_started,
                "total_processed": timer.total_processed,
                "total_displayed": timer.total_displayed,
                "total_e2e_samples": timer.total_e2e_samples,
                "processing_fps": timer.current_processing_fps,
                "display_fps": timer.current_display_fps,
                "e2e_latency_ms": timer.current_e2e_latency_ms,
                "processing_latency_ms": timer.current_processing_latency_ms,
                "avg_e2e_latency": avg_e2e_latency,
                "avg_processing_latency": avg_processing_latency,
                "processing_fps_history": list(timer.processing_fps_history),
                "display_fps_history": list(timer.display_fps_history),
                "e2e_latency_ms_history": list(timer.e2e_latency_history_ms),
                "processing_latency_ms_history": list(timer.processing_latency_history_ms),
            }

    def get_all_stats(self) -> Dict[str, dict]:
        with self.timers_lock:
            video_ids = list(self.timers.keys())
        stats = {}
        for video_id in video_ids:
            stat = self.get_stats(video_id)
            if stat:
                stats[video_id] = stat
        return stats
