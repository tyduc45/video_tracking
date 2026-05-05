# 精简实验说明

本目录只保留 2 个实验，输出统一写入项目根目录下的 `test-result`。

```powershell
python test\run_experiments.py all
```

单独运行：

```powershell
python test\run_experiments.py disorder
python test\run_experiments.py engine
```

输出内容：

- `01_乱序恢复视频对比`：两个 10 秒演示视频，`no_reorder_recovery.mp4` 和 `with_reorder_recovery.mp4`
- `04_engine_vs_pt推理速度`：`.engine` 与 `.pt` 推理速度对比图

脚本复用 `src` 中的 `YOLOInferencer`、`ByteTrackTracker` 和 `LocalVideoSource`。

批大小延迟、显存占用和分辨率区域计数实验改由 `scripts` 下的 PowerShell 脚本运行。
