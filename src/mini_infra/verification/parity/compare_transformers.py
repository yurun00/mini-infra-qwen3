"""逐步比较手写实现与 Transformers 参考实现。"""

import torch

from mini_infra.model import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    Qwen3Config,
    Qwen3Tokenizer,
    load_model,
    model_dtype_from_config,
    resolve_model_directory,
)


@torch.inference_mode()
def compare_step_by_step(
    handwritten_model,
    reference_model,
    input_ids,
    max_new_tokens,
    eos_token_id,
):
    current_input_ids = input_ids
    generated_token_ids = []
    largest_error = 0.0
    error_sum = 0.0

    for step in range(max_new_tokens):
        # 1. 手写模型单步前向：只取最后一个 Token 的 Logits 进行贪心选取。
        handwritten_logits = handwritten_model(current_input_ids)[:, -1, :]
        handwritten_token = torch.argmax(handwritten_logits, dim=-1, keepdim=True)

        # 2. Transformers 参考实现前向：同样关闭所有采样策略，执行确定性贪心解码。
        reference_logits = reference_model(input_ids=current_input_ids).logits[:, -1, :]
        reference_token = torch.argmax(reference_logits, dim=-1, keepdim=True)

        absolute_error = torch.abs(handwritten_logits - reference_logits)
        step_max_error = torch.max(absolute_error).item()
        step_mean_error = torch.mean(absolute_error).item()

        largest_error = max(largest_error, step_max_error)
        error_sum += step_mean_error

        print(
            "步骤 %02d | 手写: %5d | 参考: %5d | 最大误差: %.6f | 平均误差: %.6f"
            % (
                step + 1,
                handwritten_token.item(),
                reference_token.item(),
                step_max_error,
                step_mean_error,
            )
        )

        # 步骤中只要出现预测 Token 分歧，说明自回归历史由此产生永久性分叉，必须立即停止。
        if handwritten_token.item() != reference_token.item():
            top3_hw_vals, top3_hw_ids = torch.topk(handwritten_logits[0], k=3)
            top3_ref_vals, top3_ref_ids = torch.topk(reference_logits[0], k=3)

            print("\n[Token 不一致] 步骤 %02d:" % (step + 1))
            print("手写实现前 3 大 Logits:")
            for rank, (tid, val) in enumerate(zip(top3_hw_ids.tolist(), top3_hw_vals.tolist()), 1):
                print("  %d. token=%d, logit=%.6f" % (rank, tid, val))
            print("参考实现前 3 大 Logits:")
            for rank, (tid, val) in enumerate(zip(top3_ref_ids.tolist(), top3_ref_vals.tolist()), 1):
                print("  %d. token=%d, logit=%.6f" % (rank, tid, val))

            return {
                "tokens_match": False,
                "first_mismatch_step": step + 1,
                "largest_error": largest_error,
                "average_mean_error": error_sum / (step + 1),
                "token_ids": generated_token_ids,
            }

        next_token_id = handwritten_token.item()
        generated_token_ids.append(next_token_id)
        current_input_ids = torch.cat((current_input_ids, handwritten_token), dim=1)

        if next_token_id == eos_token_id:
            break

    return {
        "tokens_match": True,
        "first_mismatch_step": None,
        "largest_error": largest_error,
        "average_mean_error": error_sum / len(generated_token_ids),
        "token_ids": generated_token_ids,
    }


def run_compare_transformers(
    model: str = DEFAULT_MODEL_ID,
    revision: str = DEFAULT_MODEL_REVISION,
    prompt: str = "请用一句话介绍 KV Cache。",
    max_new_tokens: int = 32,
    logits_atol: float = 0.25,
) -> int:
    """与 Transformers 进行逐步对齐验证，返回退出码（0 成功，1 Token 不一致，2 误差超标，3 输入不一致）。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if max_new_tokens < 1:
        raise ValueError("--max-new-tokens 必须大于 0")
    if not torch.cuda.is_available():
        raise RuntimeError("本实验需要可用的 NVIDIA GPU")

    torch.manual_seed(0)
    device = torch.device("cuda")
    model_directory = resolve_model_directory(model, revision)
    handwritten_tokenizer = Qwen3Tokenizer(model_directory)
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
    handwritten_model = load_model(model_directory, device)

    # 强制参考实现使用 eager attention，并关闭 KV Cache，使两边执行条件一致。
    reference_model = AutoModelForCausalLM.from_pretrained(
        model_directory,
        torch_dtype=model_dtype,
        attn_implementation="eager",
    )
    reference_model.config.use_cache = False
    reference_model = reference_model.to(device).eval()

    result = compare_step_by_step(
        handwritten_model,
        reference_model,
        input_ids,
        max_new_tokens,
        handwritten_tokenizer.eos_token_id,
    )

    generated_text = handwritten_tokenizer.decode(result["token_ids"], skip_special_tokens=True)
    print("\n逐 Token 一致:", result["tokens_match"])
    print("所有步骤最大 Logits 绝对误差: %.6f" % result["largest_error"])
    print("各步骤平均 Logits 误差的均值: %.6f" % result["average_mean_error"])
    print("生成 Token IDs:", result["token_ids"])
    print("生成文本:", generated_text)

    if not result["tokens_match"]:
        print("首个不一致步骤:", result["first_mismatch_step"])
        return 1
    if result["largest_error"] > logits_atol:
        print("Token 一致，但 Logits 误差超过设定阈值。")
        return 2
    return 0


__all__ = [
    "compare_step_by_step",
    "run_compare_transformers",
]
