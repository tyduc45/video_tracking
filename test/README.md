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

- `01_乱序恢复视频对比`：两个视频，`no_reorder_recovery.mp4` 和 `with_reorder_recovery.mp4`
- `02_最小堆与batch_distrib延迟对比`：延迟折线图
- `03_batch_size影响实验`：batch_size 对 FPS 和延迟影响图
- `04_engine_vs_pt推理速度`：`.engine` 与 `.pt` 推理速度对比图

脚本复用 `src` 中的 `YOLOInferencer`、`ByteTrackTracker`、`LocalVideoSource`、`FrameData`、`BatchCollector` 和 `calculate_k_values`。
