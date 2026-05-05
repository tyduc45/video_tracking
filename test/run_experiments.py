#!/usr/bin/env python3
"""
Lean experiment runner.

Kept experiments:
- disorder: frame reorder recovery comparison videos
- engine: .engine vs .pt inference speed charts
"""

from __future__ import annotations

import argparse
import heapq
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from inference import ByteTrackTracker, YOLOInferencer
from video_source import LocalVideoSource


RESULT_ROOT = PROJECT_ROOT / "test-result"
RNG = random.Random(20260505)


@dataclass
class ClipData:
    frames: list[np.ndarray]
    detections: list[list[dict[str, Any]]]
    fps: float


def reset_dir(path: Path) -> Path:
    resolved = path.resolve()
    result_root = RESULT_ROOT.resolve()
    if resolved == result_root or result_root not in resolved.parents:
        raise ValueError(f"Refusing to reset unsafe output directory: {resolved}")
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
        candidates = [
            path for path in candidates
            if "b4" in path.stem and "imgsz1280" not in path.stem
        ] or candidates
    if not candidates:
        raise FileNotFoundError(f"model 目录下没有 {ext} 文件")
    return candidates[0]


def read_video_frames(
    video_path: Path,
    max_frames: int,
    resize_width: int | None,
) -> tuple[list[np.ndarray], float]:
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


def infer_clip(
    frames: list[np.ndarray],
    args: argparse.Namespace,
    model_path: Path | None = None,
    batch_size: int | None = None,
) -> list[list[dict[str, Any]]]:
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
    for idx in disordered:
        heapq.heappush(heap, idx)
        while heap and heap[0] == expected:
            recovered.append(heapq.heappop(heap))
            expected += 1
    while heap:
        recovered.append(heapq.heappop(heap))
    return recovered


def draw_tracking_frame(
    frame: np.ndarray,
    detections: list[dict[str, Any]],
    shown_order: int,
    original_frame_id: int,
    mode: str,
) -> np.ndarray:
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

    status = (
        "NO REORDER RECOVERY: frame order is shuffled"
        if mode == "no_recovery"
        else "WITH REORDER RECOVERY: frame order restored"
    )
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 62), (0, 0, 0), -1)
    cv2.putText(canvas, status, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    cv2.putText(
        canvas,
        f"display step {shown_order:03d} | original frame {original_frame_id:03d}",
        (12, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 0),
        2,
    )
    return canvas


def write_tracked_video(
    path: Path,
    clip: ClipData,
    order: list[int],
    session_id: str,
    mode: str,
) -> None:
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


def plot_bar(
    path: Path,
    labels: list[str],
    values: list[float],
    title: str,
    ylabel: str,
) -> None:
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


def direct_ultralytics_speed(
    model_path: Path,
    frames: list[np.ndarray],
    args: argparse.Namespace,
    batch_size: int,
) -> float:
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
    plot_bar(
        out / "engine_vs_pt_latency.png",
        ["pt", "engine"],
        [pt_time / len(frames) * 1000, engine_time / len(frames) * 1000],
        "Per-frame inference latency",
        "Latency (ms)",
    )


def run_all(args: argparse.Namespace) -> None:
    experiment_disorder(args)
    experiment_engine_vs_pt(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run selected project experiments.")
    parser.add_argument("experiment", choices=["all", "disorder", "engine"])
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--max-frames", type=int, default=90)
    parser.add_argument("--disorder-seconds", type=float, default=10.0)
    parser.add_argument("--speed-frames", type=int, default=60)
    parser.add_argument("--resize-width", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=4)
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
        "engine": experiment_engine_vs_pt,
    }
    actions[args.experiment](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
