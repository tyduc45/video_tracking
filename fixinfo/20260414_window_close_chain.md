# 主窗口关闭联动 Performance Monitor 窗口关闭机制分析

**日期**: 2026-04-14
**分支**: feature/parallel
**类型**: 机制分析 / 代码探索

---

## 现象

运行时存在两个 OpenCV 窗口：
- **主窗口** (`Video Object Detection`) — `RealtimeDisplay` 负责
- **性能监控窗口** (`Performance Monitor`) — `PerformanceWindow` 负责

当用户点击主窗口的 X 按钮（或按 Q / ESC）关闭主窗口时，性能监控窗口也会**自动随之关闭**。

---

## 关键设计要素

### 1. daemon 线程

两个显示窗口各自运行在独立的后台线程，且均设置为 `daemon=True`：

```python
# visualizer.py - RealtimeDisplay.start()
self.display_thread = threading.Thread(
    target=self._display_loop,
    name="RealtimeDisplay",
    daemon=True          # <-- daemon 线程
)

# performance_monitor.py - PerformanceWindow.start()
self.display_thread = threading.Thread(
    target=self._display_loop,
    name="PerformanceWindow",
    daemon=True          # <-- daemon 线程
)
```

daemon 线程在主进程退出时会被强制终止，是最底层的保底机制。

### 2. 每个窗口用 `cv2.getWindowProperty(WND_PROP_VISIBLE)` 自检

两个显示循环都在每次迭代中轮询自身窗口的可见性，以感知用户点击 X 按钮：

```python
# visualizer.py - RealtimeDisplay._display_loop()
window_visible = cv2.getWindowProperty(
    self.window_name, cv2.WND_PROP_VISIBLE
)
if window_visible < 1:
    self.user_quit.set()   # 标记用户主动退出
    self.stop_event.set()  # 停止自身循环
    break
```

```python
# performance_monitor.py - PerformanceWindow._display_loop()
window_visible = cv2.getWindowProperty(
    self.window_name, cv2.WND_PROP_VISIBLE
)
if window_visible < 1:
    break  # 停止自身循环
```

### 3. 窗口在创建它的线程中销毁（OpenCV 线程要求）

OpenCV 要求 `namedWindow` 和 `destroyWindow` 必须在同一个线程调用。两个显示循环都在退出循环后，在**同一线程内**销毁自己的窗口：

```python
# visualizer.py
try:
    cv2.destroyWindow(self.window_name)
except cv2.error:
    pass

# performance_monitor.py
try:
    cv2.destroyWindow(self.window_name)
except cv2.error as e:
    logger.warning(f"Failed to destroy window: {e}")
```

---

## 联动关闭的完整调用链

```
用户点击主窗口 X 按钮
        │
        ▼
RealtimeDisplay._display_loop()
  window_visible = cv2.getWindowProperty(..., WND_PROP_VISIBLE)
  → window_visible < 1
  → self.user_quit.set()       # 标记用户主动退出
  → self.stop_event.set()      # 通知主循环停止
  → break / cv2.destroyWindow("Video Object Detection")
        │
        ▼
main.py  _wait_for_pipeline() 轮询（每 100ms）
  while not pipeline.stop_event.is_set():
      if not output_handler.is_display_active():   # stop_event 已被设置
          pipeline.stop()
          break
  # finally 块必然执行：
  finally:
      output_handler.stop_display()     # 通知 RealtimeDisplay 停止（已停了也无妨）
      perf_monitor.stop()               # ← 关键：停止性能监控
        │
        ▼
PerformanceMonitor.stop()
  → self.window.stop()
        │
        ▼
PerformanceWindow.stop()
  → self.stop_event.set()      # 通知 PerformanceWindow 显示线程退出
  → self.display_thread.join(timeout=3.0)
        │
        ▼
PerformanceWindow._display_loop()
  while not self.stop_event.is_set():   # 条件不满足，退出循环
      ...
  → cv2.destroyWindow("Performance Monitor")   # 销毁性能监控窗口
```

---

## 涉及源码位置

| 步骤 | 文件 | 位置 | 说明 |
|------|------|------|------|
| 感知主窗口关闭 | `visualizer.py` | `RealtimeDisplay._display_loop()` L364-378 | `WND_PROP_VISIBLE` 轮询 |
| 设置退出标志 | `visualizer.py` | L371-372 | `user_quit.set()` + `stop_event.set()` |
| 轮询显示状态 | `main.py` | `_wait_for_pipeline()` L218-224 | 每 100ms 检查 `is_display_active()` |
| 触发停止链 | `main.py` | `_wait_for_pipeline()` finally L233-236 | `perf_monitor.stop()` |
| 传递停止信号 | `performance_monitor.py` | `PerformanceMonitor.stop()` L302-305 | 调用 `window.stop()` |
| 性能窗口退出 | `performance_monitor.py` | `PerformanceWindow.stop()` L97-106 | `stop_event.set()` + `join` |
| 销毁性能窗口 | `performance_monitor.py` | `PerformanceWindow._display_loop()` L148-151 | `cv2.destroyWindow(...)` |

---

## 设计总结

联动关闭不是通过 `cv2.destroyAllWindows()` 一次性销毁所有窗口实现的，而是通过**事件链**逐级传递停止信号：

```
用户关闭主窗口
  → RealtimeDisplay.stop_event (threading.Event)
    → main.py 主循环检测到 is_display_active() == False
      → perf_monitor.stop()
        → PerformanceWindow.stop_event (threading.Event)
          → PerformanceWindow 显示线程退出并销毁窗口
```

这样设计的好处：
- **各窗口自管理**：每个窗口只负责销毁自己，满足 OpenCV 的同线程要求
- **`finally` 保证执行**：无论是用户关窗、按 Q/ESC 还是 Ctrl+C，`finally` 块都会运行，保证性能窗口一定被关闭
- **daemon 线程兜底**：若 `stop()` 信号传递失败，进程退出时 daemon 线程会被强制终止，不会留下僵尸窗口
