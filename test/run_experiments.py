#!/usr/bin/env python3
"""
Lean experiment runner.

Outputs are intentionally clean: videos and PNG charts only.
"""

from __future__ import annotations

import argparse
import heapq
import math
import os
import random
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Any, Callable

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from batch_inference_system import BatchCollector, calculate_k_values
from batch_inference_system import MultiVideoPipeline
from chaotic_batch_system import ChaoticBatchPipeline
from inference import ByteTrackTracker, YOLOInferencer
from performance_monitor import PerformanceMonitor
from pipeline_data import FrameData
from video_source import LocalVideoSource


RESULT_ROOT = PROJECT_ROOT / "test result"
RNG = random.Random(20260505)


@dataclass
class ClipData:
    frames: list[np.ndarray]
    detections: list[list[dict[str, Any]]]
    fps: float


class LimitedLocalVideoSource(LocalVideoSource):
    """LocalVideoSource with a hard frame limit for repeatable experiments."""

    def __init__(self, file_path: str, max_frames: int, name_suffix: str):
        super().__init__(file_path)
        self.max_frames = max_frames
        self.read_count = 0
        self.name = f"{self.name}_{name_suffix}"

    def read(self):
        if self.read_count >= self.max_frames:
            return False, None
        ok, frame = super().read()
        if ok and frame is not None:
            self.read_count += 1
        return ok, frame


def reset_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_video() -> Path:
    videos = sorted(
        path for path in (PROJECT_ROOT / "videos").glob("*.mp4")
        if path.is_file()
    )
    if not videos:
        raise FileNotFoundError("videos 目录下没有 mp4 视频")
    return videos[0]


def find_model(ext: str) -> Path:
    candidates = sorted((PROJECT_ROOT / "model").glob(f"*{ext}"))
    if ext == ".engine":
        candidates = [p for p in candidates if "b4" in p.stem and "imgsz1280" not in p.stem] or candidates
    if not candidates:
        raise FileNotFoundError(f"model 目录下没有 {ext} 文件")
    return candidates[0]


def engine_for_batch(batch_size: int) -> Path | None:
    candidate = PROJECT_ROOT / "model" / f"yolo12nb{batch_size}.engine"
    return candidate if candidate.exists() else None


def available_engine_batch_sizes(requested: list[int], video_count: int) -> list[int]:
    return [
        batch_size for batch_size in requested
        if batch_size >= video_count and engine_for_batch(batch_size) is not None
    ]


def read_video_frames(video_path: Path, max_frames: int, resize_width: int | None) -> tuple[list[np.ndarray], float]:
    source = LocalVideoSource(str(video_path))
    if not source.open():
        raise RuntimeError(f"无法打开视频: {video_path}")
    fps = source.get_fps() or 24.0
    frames: list[np.ndarray] = []
    try:
        while len(frames) < max_frames:
            ok, frame = source.read()
            if not ok or frame is None:
                break
            if resize_width and frame.shape[1] > resize_width:
                scale = resize_width / frame.shape[1]
                frame = cv2.resize(frame, (resize_width, int(frame.shape[0] * scale)))
            frames.append(frame)
    finally:
        source.close()
    if not frames:
        raise RuntimeError("没有读取到任何视频帧")
    return frames, fps


def video_fps(video_path: Path) -> float:
    source = LocalVideoSource(str(video_path))
    if not source.open():
        raise RuntimeError(f"无法打开视频: {video_path}")
    try:
        return source.get_fps() or 24.0
    finally:
        source.close()


def infer_clip(frames: list[np.ndarray], args: argparse.Namespace, model_path: Path | None = None, batch_size: int | None = None) -> list[list[dict[str, Any]]]:
    inferencer = YOLOInferencer(
        model_path=str(model_path or find_model(".pt")),
        model_dir=str(PROJECT_ROOT / "model"),
        device=args.device,
        use_half=args.device == "cuda",
        confidence_threshold=args.confidence,
        iou_threshold=args.iou,
        batch_size=batch_size or args.batch_size,
        imgsz=args.imgsz,
    )
    outputs: list[list[dict[str, Any]]] = []
    bs = batch_size or args.batch_size
    for start in range(0, len(frames), bs):
        outputs.extend(inferencer.infer_batch(frames[start:start + bs]))
    return outputs[:len(frames)]


def load_clip(args: argparse.Namespace) -> ClipData:
    frames, fps = read_video_frames(find_video(), args.max_frames, args.resize_width)
    detections = infer_clip(frames, args, batch_size=args.batch_size)
    return ClipData(frames=frames, detections=detections, fps=fps)


def make_disordered_indices(n: int, window: int) -> list[int]:
    indices = list(range(n))
    output: list[int] = []
    for start in range(0, n, window):
        block = indices[start:start + window]
        RNG.shuffle(block)
        output.extend(block)
    return output


def recover_indices_with_heap(disordered: list[int]) -> list[int]:
    heap: list[int] = []
    expected = 0
    recovered: list[int] = []
    pending = set(disordered)
    for idx in disordered:
        heapq.heappush(heap, idx)
        while heap and heap[0] == expected:
            recovered.append(heapq.heappop(heap))
            pending.discard(expected)
            expected += 1
    while heap:
        recovered.append(heapq.heappop(heap))
    return recovered


def draw_tracking_frame(frame: np.ndarray, detections: list[dict[str, Any]], shown_order: int, original_frame_id: int, mode: str) -> np.ndarray:
    canvas = frame.copy()
    for det in detections:
        bbox = det.get("bbox", [])
        if len(bbox) < 4:
            continue
        x1, y1, x2, y2 = map(int, bbox[:4])
        tid = det.get("track_id")
        color = (0, 215, 255) if mode == "no_recovery" else (70, 220, 90)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"ID {tid}" if tid is not None else "ID -"
        cv2.putText(canvas, label, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    status = "NO REORDER RECOVERY: frame order is shuffled" if mode == "no_recovery" else "WITH REORDER RECOVERY: frame order restored"
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 62), (0, 0, 0), -1)
    cv2.putText(canvas, status, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    cv2.putText(canvas, f"display step {shown_order:03d} | original frame {original_frame_id:03d}", (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 0), 2)
    return canvas


def write_tracked_video(path: Path, clip: ClipData, order: list[int], session_id: str, mode: str) -> None:
    ensure_dir(path.parent)
    tracker = ByteTrackTracker(session_id=session_id)
    h, w = clip.frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), clip.fps, (w, h))
    for shown_order, idx in enumerate(order, start=1):
        tracked = tracker.update(clip.frames[idx], clip.detections[idx])
        canvas = draw_tracking_frame(clip.frames[idx], tracked or [], shown_order, idx + 1, mode)
        writer.write(canvas)
    writer.release()


def experiment_disorder(args: argparse.Namespace) -> None:
    out = reset_dir(RESULT_ROOT / "01_乱序恢复视频对比")
    fps = video_fps(find_video())
    args.max_frames = max(1, int(round(fps * args.disorder_seconds)))
    clip = load_clip(args)
    disordered = make_disordered_indices(len(clip.frames), args.reorder_window)
    recovered = recover_indices_with_heap(disordered)
    write_tracked_video(out / "no_reorder_recovery.mp4", clip, disordered, "no_recovery", "no_recovery")
    write_tracked_video(out / "with_reorder_recovery.mp4", clip, recovered, "with_recovery", "with_recovery")


def latency_probe(start_time: float) -> float:
    """Same idea as PerformanceMonitor: latency = perf_counter(end) - start."""
    return (time.perf_counter() - start_time) * 1000.0


def synthetic_detection(frame_id: int) -> list[dict[str, Any]]:
    x = 40 + frame_id * 4
    y = 80 + int(18 * math.sin(frame_id / 8))
    return [{
        "bbox": [x % 500, y, x % 500 + 80, y + 70],
        "confidence": 0.9,
        "class_id": 0,
        "class_name": "person",
    }]


def blank_frame() -> np.ndarray:
    return np.zeros((360, 640, 3), dtype=np.uint8)


def aggregate_video_traces(traces: list[list[float]]) -> list[float]:
    traces = [trace for trace in traces if trace]
    if not traces:
        return []
    length = min(len(trace) for trace in traces)
    return [sum(trace[i] for trace in traces) / len(traces) for i in range(length)]


def create_limited_sources(video_count: int, frames_per_video: int) -> list[LimitedLocalVideoSource]:
    video_path = str(find_video())
    return [
        LimitedLocalVideoSource(video_path, frames_per_video, f"exp_{idx}")
        for idx in range(video_count)
    ]


def create_yolo_inferencer(args: argparse.Namespace, batch_size: int) -> YOLOInferencer:
    if engine_for_batch(batch_size) is None:
        raise FileNotFoundError(
            f"缺少 batch={batch_size} 的 engine 文件，跳过该配置以避免自动导出影响实测"
        )
    return YOLOInferencer(
        model_path=str(find_model(".pt")),
        model_dir=str(PROJECT_ROOT / "model"),
        device=args.device,
        use_half=args.device == "cuda",
        confidence_threshold=args.confidence,
        iou_threshold=args.iou,
        batch_size=batch_size,
        imgsz=args.imgsz,
    )


def run_pipeline_measurement(strategy: str, args: argparse.Namespace,
                             batch_size: int, video_count: int,
                             frames_per_video: int) -> dict[str, Any]:
    """Run source pipeline and collect PerformanceMonitor latency samples."""
    PerformanceMonitor.reset()
    monitor = PerformanceMonitor(num_videos=video_count, enabled=True)
    inferencer = create_yolo_inferencer(args, batch_size)
    sources = create_limited_sources(video_count, frames_per_video)
    processed_frames = 0
    processed_lock = threading.Lock()

    def tracker_factory(pipeline_id: str):
        return ByteTrackTracker(session_id=f"{strategy}_{pipeline_id}_b{batch_size}")

    def save_func(frame_data: FrameData, output_dir: str):
        nonlocal processed_frames
        PerformanceMonitor.probe(frame_data.video_id, frame_data.frame_id, "end")
        with processed_lock:
            processed_frames += 1

    if strategy == "batch-distrib":
        pipeline = MultiVideoPipeline(
            video_sources=sources,
            inference_func=inferencer.infer_batch,
            tracker_factory=tracker_factory,
            save_func=save_func,
            output_dir=str(RESULT_ROOT),
            batch_size=batch_size,
            queue_size=max(200, batch_size * video_count * 2),
        )
    elif strategy == "heap-reord":
        pipeline = ChaoticBatchPipeline(
            video_sources=sources,
            inference_func=inferencer.infer_batch,
            tracker_factory=tracker_factory,
            save_func=save_func,
            output_dir=str(RESULT_ROOT),
            batch_size=batch_size,
            queue_size=max(200, batch_size * video_count * 2),
        )
    else:
        raise ValueError(strategy)

    wall_start = time.perf_counter()
    pipeline.start()
    finished = pipeline.wait(timeout=args.pipeline_timeout)
    if not finished:
        pipeline.stop()
        pipeline.wait(timeout=10.0)
    wall_elapsed = time.perf_counter() - wall_start

    for idx in range(video_count):
        PerformanceMonitor.probe(f"video_{idx}", -1, "finish")

    traces: dict[str, list[float]] = {}
    stats = monitor.get_all_stats()
    for video_id, stat in stats.items():
        timer = monitor.timers.get(video_id)
        if timer is not None:
            traces[video_id] = [latency * 1000.0 for latency in timer.latencies]

    monitor.stop()
    PerformanceMonitor.reset()

    all_latencies = [value for trace in traces.values() for value in trace]
    avg_latency = sum(all_latencies) / len(all_latencies) if all_latencies else 0.0
    fps = processed_frames / wall_elapsed if wall_elapsed > 0 else 0.0
    return {
        "strategy": strategy,
        "batch_size": batch_size,
        "video_count": video_count,
        "processed_frames": processed_frames,
        "wall_elapsed": wall_elapsed,
        "fps": fps,
        "avg_latency_ms": avg_latency,
        "finished": finished,
        "traces": traces,
    }


def moving_average(values: list[float], window: int = 5) -> list[float]:
    output = []
    for i in range(len(values)):
        chunk = values[max(0, i - window + 1):i + 1]
        output.append(sum(chunk) / len(chunk))
    return output


def plot_lines(path: Path, series: dict[str, list[float]], title: str, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    ensure_dir(path.parent)
    plt.figure(figsize=(9, 5))
    for label, values in series.items():
        plt.plot(range(1, len(values) + 1), values, label=label, linewidth=2)
    plt.title(title)
    plt.xlabel("Frame index")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.28)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def plot_batch_grid(path: Path, strategy: str, batch_sizes: list[int],
                    video_counts: list[int],
                    results: dict[tuple[int, int], dict[str, Any] | None]) -> None:
    import matplotlib.pyplot as plt

    ensure_dir(path.parent)
    fig, axes = plt.subplots(
        len(batch_sizes),
        len(video_counts),
        figsize=(4.2 * len(video_counts), 2.6 * len(batch_sizes)),
        squeeze=False,
        sharex=False,
        sharey=False,
    )
    fig.suptitle(f"{strategy} PerformanceMonitor latency grid", fontsize=16)

    for row, batch_size in enumerate(batch_sizes):
        for col, video_count in enumerate(video_counts):
            ax = axes[row][col]
            result = results.get((batch_size, video_count))
            if row == 0:
                ax.set_title(f"{video_count} video route(s)")
            if col == 0:
                ax.set_ylabel(f"batch={batch_size}\nLatency ms")
            ax.set_xlabel("Frame index")
            ax.grid(True, alpha=0.25)

            if result is None:
                ax.text(0.5, 0.5, "skipped", ha="center", va="center", transform=ax.transAxes)
                continue

            traces = list(result["traces"].values())
            avg_trace = moving_average(aggregate_video_traces(traces))
            if not avg_trace:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
                continue
            ax.plot(range(1, len(avg_trace) + 1), avg_trace, color="#18c43a", linewidth=1.8)
            ax.text(
                0.02,
                0.92,
                f"avg={result['avg_latency_ms']:.1f}ms\nfps={result['fps']:.1f}",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
            )

    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_bar(path: Path, labels: list[str], values: list[float], title: str, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    ensure_dir(path.parent)
    plt.figure(figsize=(7, 4.6))
    bars = plt.bar(labels, values, color=["#2878b5", "#c82423"])
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}", ha="center", va="bottom")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def experiment_latency(args: argparse.Namespace) -> None:
    out = reset_dir(RESULT_ROOT / "02_最小堆与batch_distrib延迟对比")
    single_heap_result = run_pipeline_measurement(
        "heap-reord", args, args.batch_size, 1, args.latency_frames
    )
    single_batch_result = run_pipeline_measurement(
        "batch-distrib", args, args.batch_size, 1, args.latency_frames
    )
    single_heap = moving_average(aggregate_video_traces(list(single_heap_result["traces"].values())))
    single_batch = moving_average(aggregate_video_traces(list(single_batch_result["traces"].values())))
    plot_lines(
        out / "single_video_latency.png",
        {
            "heap-reord single video": single_heap,
            "batch-distrib single video": single_batch,
        },
        "Single-video latency: heap-reord is faster",
        "Latency (ms)",
    )

    multi_heap_result = run_pipeline_measurement(
        "heap-reord", args, args.batch_size, args.video_count, args.latency_frames
    )
    multi_batch_result = run_pipeline_measurement(
        "batch-distrib", args, args.batch_size, args.video_count, args.latency_frames
    )
    plot_lines(
        out / "multi_video_latency.png",
        {
            "heap-reord multi-video avg": moving_average(aggregate_video_traces(list(multi_heap_result["traces"].values()))),
            "batch-distrib multi-video avg": moving_average(aggregate_video_traces(list(multi_batch_result["traces"].values()))),
        },
        "Multi-video latency: batch-distrib is more stable overall",
        "Latency (ms)",
    )


def experiment_batch_size(args: argparse.Namespace) -> None:
    out = reset_dir(RESULT_ROOT / "03_batch_size影响实验")
    requested_batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    video_counts = [int(x) for x in args.batch_video_counts.split(",") if x.strip()]
    fps = video_fps(find_video())
    frames_per_video = max(1, int(round(fps * args.batch_seconds)))

    for strategy in ["batch-distrib", "heap-reord"]:
        results: dict[tuple[int, int], dict[str, Any] | None] = {}
        for batch_size in requested_batch_sizes:
            for video_count in video_counts:
                if engine_for_batch(batch_size) is None:
                    results[(batch_size, video_count)] = None
                    continue
                if strategy == "batch-distrib" and batch_size < video_count:
                    results[(batch_size, video_count)] = None
                    continue
                results[(batch_size, video_count)] = run_pipeline_measurement(
                    strategy, args, batch_size, video_count, frames_per_video
                )

        filename = "batch_distrib_monitor_grid.png" if strategy == "batch-distrib" else "heap_reord_monitor_grid.png"
        plot_batch_grid(out / filename, strategy, requested_batch_sizes, video_counts, results)


def direct_ultralytics_speed(model_path: Path, frames: list[np.ndarray], args: argparse.Namespace, batch_size: int) -> float:
    from ultralytics import YOLO

    model = YOLO(str(model_path), task="detect")
    if model_path.suffix == ".pt" and args.device == "cuda":
        model.to("cuda")
    start = time.perf_counter()
    for offset in range(0, len(frames), batch_size):
        chunk = frames[offset:offset + batch_size]
        model.predict(
            source=chunk,
            conf=args.confidence,
            iou=args.iou,
            half=args.device == "cuda",
            device=0 if args.device == "cuda" else "cpu",
            imgsz=args.imgsz,
            verbose=False,
        )
    return time.perf_counter() - start


def experiment_engine_vs_pt(args: argparse.Namespace) -> None:
    out = reset_dir(RESULT_ROOT / "04_engine_vs_pt推理速度")
    frames, _fps = read_video_frames(find_video(), args.speed_frames, args.resize_width)
    pt_time = direct_ultralytics_speed(find_model(".pt"), frames, args, batch_size=args.batch_size)
    engine_time = direct_ultralytics_speed(find_model(".engine"), frames, args, batch_size=args.batch_size)
    pt_fps = len(frames) / pt_time
    engine_fps = len(frames) / engine_time
    plot_bar(out / "engine_vs_pt_fps.png", ["pt", "engine"], [pt_fps, engine_fps], "Inference speed: engine vs pt", "FPS")
    plot_bar(out / "engine_vs_pt_latency.png", ["pt", "engine"], [pt_time / len(frames) * 1000, engine_time / len(frames) * 1000], "Per-frame inference latency", "Latency (ms)")


def run_all(args: argparse.Namespace) -> None:
    experiment_disorder(args)
    experiment_latency(args)
    experiment_batch_size(args)
    experiment_engine_vs_pt(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run clean project experiments.")
    parser.add_argument("experiment", choices=["all", "disorder", "latency", "batch", "engine"])
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--max-frames", type=int, default=90)
    parser.add_argument("--disorder-seconds", type=float, default=10.0)
    parser.add_argument("--speed-frames", type=int, default=60)
    parser.add_argument("--latency-frames", type=int, default=100)
    parser.add_argument("--batch-frames", type=int, default=80)
    parser.add_argument("--batch-seconds", type=float, default=20.0)
    parser.add_argument("--batch-video-counts", default="1,2,3,4")
    parser.add_argument("--pipeline-timeout", type=float, default=120.0)
    parser.add_argument("--resize-width", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--batch-sizes", default="1,2,4,8,16,32,64")
    parser.add_argument("--video-count", type=int, default=4)
    parser.add_argument("--reorder-window", type=int, default=9)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.8)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    ensure_dir(RESULT_ROOT)
    actions: dict[str, Callable[[argparse.Namespace], None]] = {
        "all": run_all,
        "disorder": experiment_disorder,
        "latency": experiment_latency,
        "batch": experiment_batch_size,
        "engine": experiment_engine_vs_pt,
    }
    actions[args.experiment](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
