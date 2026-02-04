#!/usr/bin/env python3
"""
Run AITER fused_moe with a custom .co kernel using the same config as
test_moe_bit_consistency_BF16_vs_INT32.py.
"""

import argparse
import os
import shutil
import sys


KERNEL_PATH = (
    "/opt/aiter/hsa/gfx942/fmoe/silu/"
    "fmoe_bf16_pertokenFp8_g1u1_vs_silu_1tg_32x192.co"
)


def deploy_kernel(custom_kernel: str, target_kernel: str) -> None:
    if not os.path.exists(custom_kernel):
        raise FileNotFoundError(f"custom kernel not found: {custom_kernel}")
    shutil.copy2(custom_kernel, target_kernel)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run AITER fused_moe with a custom .co kernel."
    )
    parser.add_argument(
        "--kernel",
        default="/workspace/andy/poc_kl/mi300/kernel.co",
        help="Path to custom .co kernel to load.",
    )
    parser.add_argument(
        "--kernel-path",
        default=KERNEL_PATH,
        help="AITER kernel path to overwrite.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of runs to execute.",
    )
    atomic_group = parser.add_mutually_exclusive_group()
    atomic_group.add_argument(
        "--int32-atomic",
        dest="int32_atomic",
        action="store_true",
        help="Force AITER_MOE_INT32_ATOMIC=1 before running.",
    )
    atomic_group.add_argument(
        "--no-int32-atomic",
        dest="int32_atomic",
        action="store_false",
        help="Force AITER_MOE_INT32_ATOMIC unset before running.",
    )
    parser.set_defaults(int32_atomic=None)
    parser.add_argument(
        "--show-aiter-log",
        action="store_true",
        help="Show AITER kernel load logs on stderr.",
    )
    args = parser.parse_args()

    if args.int32_atomic is True:
        os.environ["AITER_MOE_INT32_ATOMIC"] = "1"
    elif args.int32_atomic is False:
        os.environ.pop("AITER_MOE_INT32_ATOMIC", None)

    if args.show_aiter_log:
        import logging

        logging.basicConfig(
            level=logging.INFO,
            format="[%(name)s] %(message)s",
            stream=sys.stderr,
        )

    deploy_kernel(args.kernel, args.kernel_path)
    print(f"Custom kernel deployed: {args.kernel} -> {args.kernel_path}")

    import torch
    from aiter.fused_moe import fused_moe
    from aiter import ActivationType, QuantType

    num_tokens = 32
    hidden_dim = 4096
    inter_dim = 192
    num_experts = 128
    top_k = 8

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    hidden = torch.randn(
        num_tokens, hidden_dim, dtype=torch.bfloat16, device="cuda"
    )
    w1 = torch.randn(
        num_experts, inter_dim * 2, hidden_dim, dtype=torch.bfloat16, device="cuda"
    ).to(torch.float8_e4m3fnuz)
    w2 = torch.randn(
        num_experts, hidden_dim, inter_dim, dtype=torch.bfloat16, device="cuda"
    ).to(torch.float8_e4m3fnuz)

    scores = torch.randn(num_tokens, num_experts, dtype=torch.float32, device="cuda")
    weights = torch.softmax(scores, dim=-1)
    topk_weights, topk_ids = torch.topk(weights, k=top_k, dim=-1)

    w1_scale = torch.ones(
        num_experts, inter_dim * 2, 1, dtype=torch.float32, device="cuda"
    )
    w2_scale = torch.ones(
        num_experts, hidden_dim, 1, dtype=torch.float32, device="cuda"
    )

    outputs = []
    for _ in range(args.runs):
        output = fused_moe(
            hidden.clone(),
            w1.clone(),
            w2.clone(),
            topk_weights.to(torch.float32).clone(),
            topk_ids.to(torch.int32).clone(),
            quant_type=QuantType.per_Token,
            w1_scale=w1_scale.clone(),
            w2_scale=w2_scale.clone(),
            activation=ActivationType.Silu,
        )
        outputs.append(output[0, :8].float().cpu().tolist())

    print("Output (token 0, first 8 values) per run:")
    for i, out in enumerate(outputs, start=1):
        print(f"  Run {i:2d}: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
