# 精简实验说明

本目录只保留 4 个实验，输出统一写入项目根目录下的 `test result`。

```powershell
python test\run_experiments.py all
```

单独运行：

```powershell
python test\run_experiments.py disorder
python test\run_experiments.py latency
python test\run_experiments.py batch
python test\run_experiments.py engine
```

输出内容：

- `01_乱序恢复视频对比`：两个 10 秒演示视频，`no_reorder_recovery.mp4` 和 `with_reorder_recovery.mp4`
- `02_最小堆与batch_distrib延迟对比`：单视频与多视频两张延迟折线图
- `03_batch_size影响实验`：直接运行源码 `PerformanceMonitor` 实测 20 秒视频延迟；按策略输出两张矩阵图，横向为视频路数，纵向为 batch_size
- `04_engine_vs_pt推理速度`：`.engine` 与 `.pt` 推理速度对比图

脚本复用 `src` 中的 `YOLOInferencer`、`ByteTrackTracker`、`LocalVideoSource`、`FrameData`、`BatchCollector` 和 `calculate_k_values`。

延迟结论按实测现象组织：优先队列 `heap-reord` 在单视频下延迟更低；多视频并发时 `batch-distrib` 的跨视频平均延迟更稳定，更适合作为综合策略。
