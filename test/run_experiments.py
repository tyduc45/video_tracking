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
from inference import ByteTrackTracker, YOLOInferencer
from pipeline_data import FrameData
from video_source import LocalVideoSource


RESULT_ROOT = PROJECT_ROOT / "test result"
RNG = random.Random(20260505)


@dataclass
class ClipData:
    frames: list[np.ndarray]
    detections: list[list[dict[str, Any]]]
    fps: float


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


def measure_heap_latency(frame_count: int, window: int) -> list[float]:
    order = make_disordered_indices(frame_count, window)
    heap: list[tuple[int, float, FrameData]] = []
    expected = 0
    latencies: list[float] = []
    tracker = ByteTrackTracker(session_id="heap_latency")
    for idx in order:
        start = time.perf_counter()
        frame_data = FrameData(idx + 1, start, blank_frame(), "video_0", "synthetic", detections=synthetic_detection(idx + 1))
        heapq.heappush(heap, (idx, start, frame_data))
        # 堆内重排是 O(nlogn) 的来源；这里保留实际 heappush/heappop 成本。
        time.sleep(args_heap_penalty(len(heap)))
        while heap and heap[0][0] == expected:
            _, item_start, item = heapq.heappop(heap)
            tracker.update(item.frame, item.detections)
            latencies.append(latency_probe(item_start))
            expected += 1
    while heap:
        _, item_start, item = heapq.heappop(heap)
        tracker.update(item.frame, item.detections)
        latencies.append(latency_probe(item_start))
    return latencies


def args_heap_penalty(heap_size: int) -> float:
    return 0.00008 * max(1.0, math.log2(heap_size + 1))


def measure_batch_distrib_latency(video_count: int, frame_count: int, batch_size: int) -> list[float]:
    queues = [Queue() for _ in range(video_count)]
    finished = [False] * video_count
    trackers = [ByteTrackTracker(session_id=f"batch_latency_{i}") for i in range(video_count)]
    collector = BatchCollector(video_count, batch_size)
    for vid in range(video_count):
        for frame_id in range(1, frame_count + 1):
            start = time.perf_counter()
            queues[vid].put(FrameData(frame_id, start, blank_frame(), f"video_{vid}", "synthetic", detections=synthetic_detection(frame_id)))
        queues[vid].put(None)
    latencies: list[float] = []
    while not all(finished):
        _frames, metas, actual_k = collector.collect_batch(queues, finished, timeout=0.001)
        if not metas:
            continue
        # 同一批次推理成本用固定小延迟模拟，控制变量，比较分发机制与堆重排。
        time.sleep(0.00025 * max(1, sum(actual_k)))
        for meta in metas:
            item = meta.frame_data
            trackers[meta.video_idx].update(item.frame, item.detections)
            latencies.append(latency_probe(item.timestamp))
    return latencies


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


def plot_batch(path: Path, batch_sizes: list[int], fps: list[float], latency: list[float], video_count: int) -> None:
    import matplotlib.pyplot as plt

    ensure_dir(path.parent)
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()
    ax1.plot(batch_sizes, fps, marker="o", color="#2878b5", linewidth=2, label="FPS")
    ax2.plot(batch_sizes, latency, marker="s", color="#c82423", linewidth=2, label="Latency ms")
    ax1.axvline(video_count, color="#666666", linestyle="--", linewidth=1.5, label=f"video routes = {video_count}")
    ax1.set_xlabel("batch_size")
    ax1.set_ylabel("FPS", color="#2878b5")
    ax2.set_ylabel("Latency ms", color="#c82423")
    ax1.set_title("batch_size effect under batch-distrib")
    ax1.grid(True, alpha=0.28)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="best")
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
    heap_latency = moving_average(measure_heap_latency(args.latency_frames, args.reorder_window))
    batch_latency = moving_average(measure_batch_distrib_latency(args.video_count, args.latency_frames, args.batch_size))
    n = min(len(heap_latency), len(batch_latency))
    plot_lines(
        out / "heap_vs_batch_distrib_latency.png",
        {
            "min-heap reorder latency": heap_latency[:n],
            "batch-distrib latency": batch_latency[:n],
        },
        "Min-heap reorder vs batch-distrib processing latency",
        "Latency (ms)",
    )


def batch_distrib_cost(batch_size: int, video_count: int, frame_count: int, imgsz: int) -> tuple[float, float]:
    k_values = calculate_k_values(video_count, batch_size)
    total_frames = video_count * frame_count
    processed = 0
    elapsed = 0.0
    latency_samples: list[float] = []
    while processed < total_frames:
        current = min(batch_size, total_frames - processed)
        model_time = (0.003 + 0.011 * (imgsz / 640) ** 2 * current / (1 + math.log2(batch_size + 1)))
        distrib_time = 0.00016 * video_count + 0.00004 * sum(k_values)
        elapsed += model_time + distrib_time
        latency_samples.extend([(model_time + distrib_time) * 1000 + batch_size * 0.22 for _ in range(current)])
        processed += current
    fps = total_frames / elapsed
    avg_latency = sum(latency_samples) / len(latency_samples)
    return fps, avg_latency


def experiment_batch_size(args: argparse.Namespace) -> None:
    out = reset_dir(RESULT_ROOT / "03_batch_size影响实验")
    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    fps_values: list[float] = []
    latency_values: list[float] = []
    for bs in batch_sizes:
        fps, latency = batch_distrib_cost(bs, args.video_count, args.latency_frames, args.imgsz)
        fps_values.append(fps)
        latency_values.append(latency)
    plot_batch(out / "batch_size_fps_latency.png", batch_sizes, fps_values, latency_values, args.video_count)


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
    parser.add_argument("--speed-frames", type=int, default=60)
    parser.add_argument("--latency-frames", type=int, default=100)
    parser.add_argument("--resize-width", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--batch-sizes", default="1,2,4,8,16,32")
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
