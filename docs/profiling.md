# 算子性能剖析与诊断指南 (Profiling & Trace Guide)

在进行 LLM 推理引擎性能优化时，仅看吞吐或延迟指标往往难以准确定位瓶颈。本项目内建了轻量级的 PyTorch Profiler 剖析打点与时序导出机制，支持快速定位 Kernel 耗时、CPU 调度空泡与显存占用瓶颈。

---

## 一、运行性能剖析与导出 Trace

使用 `mini-infra bench batch case` 附带 `--profile-mode full-profile`（或 `extra-profile`）进行采样分析，可同时生成结构化性能 JSON 并导出底层的 Chrome / TensorBoard Trace：

```bash
# 运行轻量压测进行算子耗时诊断，落盘诊断 JSON 并导出 Trace 日志
mini-infra bench batch case \
  --tag debug \
  --prompt-lens 512 \
  --max-new-tokens 32 \
  --repeats 1 \
  --profile-mode full-profile \
  --tensorboard-dir tmp/tb_trace \
  --output tmp/profile.json
```

---

## 二、Trace 可视化与瓶颈分析

导出的 Trace 文件保存在 `tmp/tb_trace/` 目录下，包含标准的 PyTorch Profiler 产物（`*.pt.trace.json`）。建议采用以下两种方式进行可视化分析：

### 方式一：更轻量流畅的方案 —— Perfetto / Chrome Tracing（推荐）

在大尺寸 trace 或复杂模型调用栈下，Perfetto 比 TensorBoard 更轻量，支持秒级加载与 60fps 流畅交互：

1. **获取 Trace 文件**：
   从远程测试机的 `tmp/tb_trace/*.pt.trace.json` 下载到本地（VS Code / Antigravity IDE 用户可直接在左侧文件浏览器中右键点击下载）。
2. **在浏览器中打开**：
   访问 [https://ui.perfetto.dev/](https://ui.perfetto.dev/)（或 Chrome 浏览器内打开 `chrome://tracing`）。
3. **加载与交互**：
   点击左侧 **"Open trace file"**（或直接将 `*.pt.trace.json` 拖入浏览器窗口）。
4. **分析重点**：
   * 使用键盘 `W/A/S/D` 进行缩放与平移。
   * 查看 **CUDA Stream** 上的底层 Kernel 执行时序（如 `gemm`、`vectorized_elementwise_kernel`）。
   * 观察是否存在 **CPU 调度延迟** 导致的 GPU 空置空泡（GPU Bubbles）。

---

### 方式二：使用 TensorBoard (PyTorch Profiler 插件)

TensorBoard 提供了算子层面的统计表格、显存占用折线图以及 GPU 利用率视图：

1. **在远程实例上启动 TensorBoard 服务**：
   ```bash
   tensorboard --logdir tmp/tb_trace --port 6006
   ```

2. **端口转发至本地（任选一种）**：
   * **IDE 原生端口转发（推荐）**：在 VS Code / Antigravity IDE 底部状态栏的 **"PORTS (端口)"** 面板添加端口 `6006`，IDE 会自动完成安全隧道转发。
   * **SSH 端口转发**：
     ```bash
     ssh -N -L 6006:localhost:6006 <user>@<remote_ip>
     ```
   * **GCP Cloud SDK 端口转发**：
     ```bash
     gcloud compute ssh <instance_name> --zone=<zone> -- -N -L 6006:localhost:6006
     ```

3. **浏览器查看**：
   本地浏览器打开 `http://localhost:6006`，并在页面右上角插件下拉菜单中切换至 **"PyTorch Profiler"** 视图。

---

## 三、关键诊断指标与优化思路

在分析 `mini-infra bench batch case` 的 Profiling 报告或可视化视图时，重点关注以下四大维度：

| 诊断维度 | 核心观察指标 | 优化指引 |
| :--- | :--- | :--- |
| **模块归因** | Attention vs. MLP vs. RMSNorm 耗时占比 | • 若 Attention 占比高：引入 KV Cache、FlashAttention / SDPA 融合算子<br>• 若 RMSNorm 占比高：考虑 Fused RMSNorm（融合残差与归一化） |
| **算子与访存** | GEMM 矩阵乘 vs. Elementwise / Memory 搬运占比 | • 若 Elementwise 与搬运占比高：中间临时张量多，需横向/纵向算子融合减少 HBM 读写回 |
| **硬件利用率** | MBU（显存带宽利用率）与 MFU（计算利用率） | • Decode 阶段为典型 Memory-bound，理想 MBU 在 60%~80% 之间<br>• 若 MBU 极低且 GPU 忙碌：存在大量低算力开销的重复计算 |
| **显存足迹** | 动态激活显存 vs. PyTorch Allocator 预留与碎片 | • 长序列或多批处理下若碎片大：引入分页式内存管理（PagedAttention） |
