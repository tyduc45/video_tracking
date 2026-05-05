#!/usr/bin/env python3
"""
Plot latency CSV files from the batch-size experiment.

Input CSV file names are expected to look like:
latency_source3_bs16_batch-distrib_20260505_200847.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_DIR = PROJECT_ROOT / "test result" / "03_batch_size影响实验"

FILENAME_RE = re.compile(
    r"^latency_source(?P<videos>\d+)_bs(?P<batch>\d+)_(?P<strategy>.+?)_"
    r"(?P<timestamp>\d{8}_\d{6})\.csv$"
)


@dataclass(frozen=True)
class LatencyRun:
    path: Path
    videos: int
    batch: int
    strategy: str
    timestamp: str


def parse_run(path: Path) -> LatencyRun | None:
    match = FILENAME_RE.match(path.name)
    if not match:
        return None

    return LatencyRun(
        path=path,
        videos=int(match.group("videos")),
        batch=int(match.group("batch")),
        strategy=match.group("strategy"),
        timestamp=match.group("timestamp"),
    )


def discover_latest_runs(input_dir: Path) -> dict[tuple[str, int, int], LatencyRun]:
    latest: dict[tuple[str, int, int], LatencyRun] = {}
    for path in input_dir.glob("latency_*.csv"):
        run = parse_run(path)
        if run is None:
            continue

        key = (run.strategy, run.videos, run.batch)
        previous = latest.get(key)
        if previous is None or run.timestamp > previous.timestamp:
            latest[key] = run

    return latest


def read_latency(path: Path) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            idx_text = row.get("idx") or row.get("frameid")
            latency_text = row.get("latency_ms")
            if not idx_text or not latency_text:
                continue
            xs.append(int(float(idx_text)))
            ys.append(float(latency_text))
    return xs, ys


def plot_strategy_grid(strategy: str, runs: list[LatencyRun], output_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    batches = sorted({run.batch for run in runs}, reverse=True)
    video_counts = sorted({run.videos for run in runs})
    by_cell = {(run.videos, run.batch): run for run in runs}

    fig_width = max(4.0 * len(video_counts), 6.0)
    fig_height = max(2.8 * len(batches), 4.0)
    fig, axes = plt.subplots(
        len(batches),
        len(video_counts),
        figsize=(fig_width, fig_height),
        squeeze=False,
        sharex=False,
        sharey=False,
    )
    fig.suptitle(f"{strategy} latency grid", fontsize=16)

    for row, batch in enumerate(batches):
        for col, videos in enumerate(video_counts):
            ax = axes[row][col]
            run = by_cell.get((videos, batch))

            ax.grid(True, alpha=0.25)
            if row == 0:
                ax.set_title(f"{videos} video source(s)")
            if col == 0:
                ax.set_ylabel(f"batch={batch}\nlatency_ms")
            if row == len(batches) - 1:
                ax.set_xlabel("idx")

            if run is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
                continue

            xs, ys = read_latency(run.path)
            if not xs:
                ax.text(0.5, 0.5, "empty", ha="center", va="center", transform=ax.transAxes)
                continue

            ax.plot(xs, ys, linewidth=1.4)
            avg = sum(ys) / len(ys)
            ax.text(
                0.02,
                0.95,
                f"avg={avg:.1f}ms\nn={len(ys)}",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none"},
            )

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_strategy = re.sub(r"[^A-Za-z0-9_.-]+", "_", strategy).strip("_")
    output_path = output_dir / f"{safe_strategy}_latency_grid.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot latency grids from record-lat CSV files.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULT_DIR)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    latest = discover_latest_runs(input_dir)
    if not latest:
        print(f"No latency CSV files found in {input_dir}")
        return 1

    by_strategy: dict[str, list[LatencyRun]] = {}
    for run in latest.values():
        by_strategy.setdefault(run.strategy, []).append(run)

    for strategy in sorted(by_strategy):
        output_path = plot_strategy_grid(strategy, by_strategy[strategy], output_dir)
        print(f"Wrote {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
