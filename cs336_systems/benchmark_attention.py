import argparse
import timeit
from collections.abc import Callable

import torch
from cs336_basics.model import scaled_dot_product_attention

BATCH_SIZE = 8
D_MODEL_VALUES = [16, 32, 64, 128]
SEQ_LEN_VALUES = [256, 1024, 4096, 8192, 16384]
WARMUP_STEPS = 5
NUM_STEPS = 100


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true", help="also benchmark torch.compile(attention)")
    return parser.parse_args()


def benchmark_config(attention: Callable, seq_len: int, d_model: int, device: torch.device) -> tuple[float, float, float]:
    shape = (BATCH_SIZE, seq_len, d_model)
    Q = torch.randn(shape, device=device, requires_grad=True)
    K = torch.randn(shape, device=device, requires_grad=True)
    V = torch.randn(shape, device=device, requires_grad=True)
    grad_output = torch.randn(shape, device=device)

    def zero_grads():
        Q.grad = K.grad = V.grad = None

    for _ in range(WARMUP_STEPS):
        attention(Q, K, V).backward(grad_output)
        zero_grads()
    torch.cuda.synchronize()

    forward_time = 0.0
    for _ in range(NUM_STEPS):
        start_time = timeit.default_timer()
        attention(Q, K, V)
        torch.cuda.synchronize()
        forward_time += timeit.default_timer() - start_time

    backward_time = 0.0
    memory_before_backward = 0.0
    for _ in range(NUM_STEPS):
        output = attention(Q, K, V)
        torch.cuda.synchronize()
        memory_before_backward = torch.cuda.memory_allocated()

        start_time = timeit.default_timer()
        output.backward(grad_output)
        torch.cuda.synchronize()
        backward_time += timeit.default_timer() - start_time
        zero_grads()

    return forward_time / NUM_STEPS * 1000, backward_time / NUM_STEPS * 1000, memory_before_backward / 1024**2


def main():
    args = parse_args()
    device = torch.device("cuda")
    implementations = ["eager", "compiled"] if args.compile else ["eager"]

    print(f"{'seq_len':<10} | {'d_model':<10} | {'impl':<10} | {'forward (ms)':<15} | {'backward (ms)':<15} | {'memory before backward (MiB)':<30}")

    for seq_len in SEQ_LEN_VALUES:
        for d_model in D_MODEL_VALUES:
            for impl in implementations:
                torch._dynamo.reset()
                attention = torch.compile(scaled_dot_product_attention) if impl == "compiled" else scaled_dot_product_attention
                try:
                    forward_ms, backward_ms, memory_mib = benchmark_config(attention, seq_len, d_model, device)
                    print(f"{seq_len:<10} | {d_model:<10} | {impl:<10} | {forward_ms:<15.3f} | {backward_ms:<15.3f} | {memory_mib:<30.1f}")
                except torch.cuda.OutOfMemoryError:
                    print(f"{seq_len:<10} | {d_model:<10} | {impl:<10} | {'OOM':<15} | {'OOM':<15} | {'OOM':<30}")
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
