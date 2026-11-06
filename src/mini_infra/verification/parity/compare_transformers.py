"""逐步比较手写实现与 Transformers 参考实现。"""

import torch

from mini_infra.model import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    Qwen3Config,
    model_dtype_from_config,
)
from mini_infra.verification.parity._common import (
    build_environment,
    finalize_stepwise_compare,
    prepare_cuda_model,
    run_stepwise_greedy_compare,
)


@torch.inference_mode()
def compare_transformers_step_by_step(
    handwritten_model,
    reference_model,
    input_ids,
    max_new_tokens,
    eos_token_id,
):
    current_input_ids = input_ids

    def step_fn():
        nonlocal current_input_ids
        # 1. 手写模型单步前向：只取最后一个 Token 的 Logits 进行贪心选取（返回的历史键值这里用不到）。
        handwritten_logits = handwritten_model(current_input_ids)[0][:, -1, :]
        # 2. Transformers 参考实现前向：同样关闭所有采样策略，执行确定性贪心解码。
        reference_logits = reference_model(input_ids=current_input_ids).logits[:, -1, :]

        next_token = torch.argmax(handwritten_logits, dim=-1, keepdim=True)
        current_input_ids = torch.cat((current_input_ids, next_token), dim=1)
        return reference_logits, handwritten_logits, next_token.item(), None

    return run_stepwise_greedy_compare(
        step_fn,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
        reference_name="参考实现",
        candidate_name="手写实现",
    )


def execute_compare_transformers(
    model: str = DEFAULT_MODEL_ID,
    revision: str = DEFAULT_MODEL_REVISION,
    prompt: str = "请用一句话介绍 KV Cache。",
    max_new_tokens: int = 32,
    logits_atol: float = 0.25,
    output: str | None = None,
) -> int:
    """与 Transformers 进行逐步对齐验证，返回退出码（0 成功，1 Token 不一致，2 误差超标，3 输入不一致）。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if max_new_tokens < 1:
        raise ValueError("--max-new-tokens 必须大于 0")

    device, model_directory, handwritten_tokenizer, handwritten_model = prepare_cuda_model(model, revision)
    reference_tokenizer = AutoTokenizer.from_pretrained(model_directory)

    # 模型对照开始前，先证明手写 Tokenizer 构造了完全相同的输入。
    handwritten_ids = handwritten_tokenizer.encode_chat_prompt(prompt)
    messages = [{"role": "user", "content": prompt}]
    reference_text = reference_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    reference_ids = reference_tokenizer(
        reference_text,
        add_special_tokens=False,
    ).input_ids
    if handwritten_ids != reference_ids:
        print("手写 Tokenizer 与 Transformers 输入不一致")
        print("手写 Token IDs:", handwritten_ids)
        print("参考 Token IDs:", reference_ids)
        return 3

    print("Tokenizer 输入一致，共 %d 个 Token" % len(handwritten_ids))
    input_ids = torch.tensor([handwritten_ids], dtype=torch.long, device=device)

    config = Qwen3Config.from_model_directory(model_directory)
    model_dtype = model_dtype_from_config(config)

    environment = build_environment(
        device,
        model=model,
        revision=revision,
        dtype=config.torch_dtype,
        prompt_tokens=len(handwritten_ids),
        max_new_tokens=max_new_tokens,
        logits_atol=logits_atol,
        decoding="greedy",
        thinking=False,
    )

    # 强制参考实现使用 eager attention，并关闭 KV Cache，使两边执行条件一致。
    reference_model = AutoModelForCausalLM.from_pretrained(
        model_directory,
        torch_dtype=model_dtype,
        attn_implementation="eager",
    )
    reference_model.config.use_cache = False
    reference_model = reference_model.to(device).eval()

    result = compare_transformers_step_by_step(
        handwritten_model,
        reference_model,
        input_ids,
        max_new_tokens,
        handwritten_tokenizer.eos_token_id,
    )

    generated_text = handwritten_tokenizer.decode(result["token_ids"], skip_special_tokens=True)
    return finalize_stepwise_compare(
        result, generated_text, logits_atol=logits_atol, output=output, environment=environment
    )


__all__ = [
    "compare_transformers_step_by_step",
    "execute_compare_transformers",
]
