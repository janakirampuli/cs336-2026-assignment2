import argparse
import math
import timeit

import cs336_basics.model
import numpy as np
import torch
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy, softmax
from cs336_basics.optimizer import AdamW
from torch import einsum
from torch.cuda import nvtx


def annotated_scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    with nvtx.range("scaled dot product attention"):
        d_k = K.size(-1)

        with nvtx.range("computing attention scores"):
            attention_scores = einsum("... q d, ... k d -> ... q k", Q, K) / math.sqrt(d_k)

        if mask is not None:
            with nvtx.range("masking"):
                attention_scores = attention_scores.masked_fill(~mask, -torch.inf)

        with nvtx.range("computing softmax"):
            attention_weights = softmax(attention_scores, dim=-1)

        with nvtx.range("final matmul"):
            output = einsum("... q k, ... k v -> ... q v", attention_weights, V)

    return output


def parse_args():
    parser = argparse.ArgumentParser()

    # model hyperparams
    parser.add_argument("--d_model", type=int, default=512, help="embedding dimension")
    parser.add_argument("--num_layers", type=int, default=4, help="number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=16, help="number of attention heads")
    parser.add_argument("--d_ff", type=int, default=1344, help="feed forward dimension")
    parser.add_argument("--vocab_size", type=int, default=10000, help="vocabulary size")
    parser.add_argument("--context_length", type=int, default=512, help="maximum sequence length")

    # benchmarking params
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--warmup_steps", type=int, default=5, help="number of warmup steps")
    parser.add_argument("--num_steps", type=int, default=10, help="number of steps to measure")
    parser.add_argument("--mode", type=str, default="forward", choices=["forward", "backward", "train"], help="benchmarking mode")

    # device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="device to use (cuda, mps, cpu)")

    # mixed precision
    parser.add_argument("--autocast_dtype", type=str, default="float32", choices=["float32", "bfloat16"], help="compute precision")

    return parser.parse_args()


def main():
    args = parse_args()

    cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

    device = torch.device(args.device)
    print(f"using device: {device}")
    print(f"using {args.autocast_dtype}")

    def autocast():
        return torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.autocast_dtype == "bfloat16")

    model_config = {
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "num_layers": args.num_layers,
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "d_ff": args.d_ff,
    }

    print(f"TransformerLM config: {model_config}")

    model = BasicsTransformerLM(**model_config)
    model.to(device)
    model.train()

    print(f"model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.3f}M")

    inputs = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)
    targets = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)

    if args.mode == "train":
        optimizer = AdamW(params=model.parameters())

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

    def run_step(step_name):
        with nvtx.range(step_name):
            if args.mode == "backward":
                model.zero_grad(set_to_none=True)

            with autocast():
                with nvtx.range("forward"):
                    logits = model(inputs)
                if args.mode != "forward":
                    loss = cross_entropy(logits.float(), targets)

            if args.mode != "forward":
                with nvtx.range("backward"):
                    loss.backward()

            if args.mode == "train":
                with nvtx.range("optimizer"):
                    optimizer.step()
                    optimizer.zero_grad()

            synchronize()

    if args.warmup_steps > 0:
        print(f"performing warmup for {args.warmup_steps} steps...")
        with nvtx.range("warmup"):
            for i in range(args.warmup_steps):
                run_step(step_name=f"warmup_step_{i}")

    print("benchmarking...")
    step_times = []

    with nvtx.range("measure"):
        for i in range(args.num_steps):
            start_time = timeit.default_timer()
            run_step(step_name=f"measure_step_{i}")
            end_time = timeit.default_timer()
            step_times.append(end_time - start_time)

    step_times = np.array(step_times)
    mean_time = np.mean(step_times)
    std_time = np.std(step_times)

    print(f"results ({args.mode}):")
    print(f"mean time: {mean_time * 1000:.2f} ms")
    print(f"std dev: {std_time * 1000:.2f} ms")


if __name__ == "__main__":
    main()
