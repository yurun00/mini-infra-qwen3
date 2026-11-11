---
name: Feature Benchmark
about: 对比评估基线与目标推理/调度模式在吞吐、时延与显存等维度的性能表现
title: "[Bench] Performance benchmark comparison between <target> and <baseline> modes"
labels: ["type:benchmark"]
assignees: ""
---

## 📌 Overview

This issue summarizes the performance benchmark comparison between the **`<Baseline Mode>`** and **`<Target Mode>`** inference modes for `Qwen/Qwen3-0.6B`.

The objective is to quantify generation throughput scaling, latency characteristics (TTFT / ITL / Makespan), and GPU memory / resource utilization trade-offs across the following core dimensions:
1. **`<Dimension 1>`**: 
2. **`<Dimension 2>`**: 
3. **`<Dimension 3>`**: 

---

## 🖥️ Benchmark Environment

- **Model**: `Qwen/Qwen3-0.6B`
- **Revision**: `c1899de289a04d12100db370d81485cdf75e47ca`
- **Precision / Dtype**: `torch.bfloat16`
- **GPU**: NVIDIA L4 (24GB VRAM, PCIe Gen4)
- **CUDA / Driver**: CUDA 13.0 / Driver 580.178.04
- **PyTorch**: 2.14.1+cu130
- **Decoding Configuration**: Greedy decoding, forced output budget, EOS disabled
- **Test Workloads**:
  - **Suite 1 (`<suite-name-1>`)**: 
  - **Suite 2 (`<suite-name-2>`)**: 
  - **Suite 3 (`<suite-name-3>`)**: 

---

## 🚀 How to Reproduce

```bash
# 1. Benchmark Suite 1 (<description>)
mini-infra bench sweep --engine scheduler --tag dev --suite <suite-name-1>

# 2. Benchmark Suite 2 (<description>)
mini-infra bench sweep --engine scheduler --tag dev --suite <suite-name-2>

# 3. Benchmark Suite 3 (<description>)
mini-infra bench sweep --engine scheduler --tag dev --suite <suite-name-3>

# 4. (Optional) Run individual parameterized case
mini-infra bench scheduler case --tag dev --batching-method <method> --max-running-requests 4 --request-count 8 --prompt-length 128 --output-budget 32
```

