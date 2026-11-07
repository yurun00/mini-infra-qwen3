"""Qwen3 模型目录解析与 safetensors 权重加载。

将文件 / 网络 I/O（Hugging Face Hub 下载、磁盘权重读取与设备搬运）与 qwen3.py 的纯张量计算图解耦。
"""

from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from mini_infra.model.config import DEFAULT_MODEL_REVISION, Qwen3Config, model_dtype_from_config
from mini_infra.model.qwen3 import Qwen3ForCausalLM


def resolve_model_directory(model_name_or_path: str, revision: str = DEFAULT_MODEL_REVISION) -> str:
    """返回本地模型目录；本地路径不存在时从 Hugging Face Hub 下载固定 revision。"""
    local_path = Path(model_name_or_path).expanduser()
    if local_path.is_dir():
        return str(local_path.resolve())

    return snapshot_download(
        repo_id=model_name_or_path,
        revision=revision,
        max_workers=2,
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "merges.txt",
            "vocab.json",
        ],
    )


def load_model(model_directory: str | Path, device: torch.device, dtype: torch.dtype | None = None) -> Qwen3ForCausalLM:
    """从本地目录的 model.safetensors 构造并加载 Qwen3ForCausalLM（eval 态）。"""
    config = Qwen3Config.from_model_directory(model_directory)
    model = Qwen3ForCausalLM(config)

    weight_path = Path(model_directory) / "model.safetensors"
    if not weight_path.is_file():
        raise FileNotFoundError("没有找到模型权重: " + str(weight_path))

    state_dict = load_file(str(weight_path), device="cpu")
    load_result = model.load_state_dict(state_dict, strict=False)

    allowed_missing_keys = {"lm_head.weight"}
    unexpected_missing = set(load_result.missing_keys) - allowed_missing_keys
    if unexpected_missing or load_result.unexpected_keys:
        raise RuntimeError(
            "权重名称不匹配，missing="
            + str(sorted(unexpected_missing))
            + ", unexpected="
            + str(sorted(load_result.unexpected_keys))
        )

    model.tie_weights()
    model_dtype = model_dtype_from_config(config) if dtype is None else dtype
    if model_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("不支持的模型 dtype: " + str(model_dtype))

    # inv_freq 保持 float32 精度，避免 model.to(dtype=model_dtype) 将旋转频率降为低精度。
    rotary_inv_freq = model.model.rotary_emb.inv_freq.clone()
    model = model.to(device=device, dtype=model_dtype)
    model.model.rotary_emb.inv_freq = rotary_inv_freq.to(device)
    model.eval()
    return model


__all__ = [
    "load_model",
    "resolve_model_directory",
]
