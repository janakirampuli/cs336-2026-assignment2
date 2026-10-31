import argparse
import timeit

import numpy as np
import torch
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from torch.utils.checkpoint import checkpoint


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

    # memory profiling
    parser.add_argument("--memory_snapshot", type=str, default=None, help="save a CUDA memory snapshot of the timed steps to this .pickle path")

    # gradient checkpointing
    parser.add_argument("--checkpoint_every", type=int, default=0, help="wrap every k transformer layers in one checkpoint (0 = no checkpointing)")

    # torch.compile
    parser.add_argument("--compile", action="store_true", help="run the model through torch.compile")

    return parser.parse_args()


def run_layers(layers: torch.nn.ModuleList, x: torch.Tensor) -> torch.Tensor:
    for layer in layers:
        x = layer(x)
    return x


def checkpointed_forward(model: BasicsTransformerLM, inputs: torch.Tensor, checkpoint_every: int) -> torch.Tensor:
    x = model.token_embeddings(inputs)
    for start in range(0, len(model.layers), checkpoint_every):
        segment = model.layers[start : start + checkpoint_every]
        x = checkpoint(run_layers, segment, x, use_reentrant=False)
    x = model.ln_final(x)
    x = model.lm_head(x)
    return x


def main():
    args = parse_args()

    device = torch.device(args.device)
    if args.memory_snapshot and device.type != "cuda":
        raise ValueError("--memory_snapshot needs a cuda device")
    if args.compile and args.checkpoint_every > 0:
        raise ValueError("--compile and --checkpoint_every can't be combined")
    print(f"using device: {device}")
    print(f"using {args.autocast_dtype}")
    if args.checkpoint_every > 0:
        print(f"checkpointing every {args.checkpoint_every} layers")
    if args.compile:
        print("using torch.compile")

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

    if args.compile:
        model = torch.compile(model)

    inputs = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)
    targets = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)

    if args.mode == "train":
        optimizer = AdamW(params=model.parameters())

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

    def run_step():
        if args.mode == "backward":
            model.zero_grad(set_to_none=True)

        with autocast():
            if args.checkpoint_every > 0:
                logits = checkpointed_forward(model, inputs, args.checkpoint_every)
            else:
                logits = model(inputs)
            if args.mode != "forward":
                loss = cross_entropy(logits.float(), targets)

        if args.mode != "forward":
            loss.backward()

        if args.mode == "train":
            optimizer.step()
            optimizer.zero_grad()

        synchronize()

    if args.warmup_steps > 0:
        print(f"performing warmup for {args.warmup_steps} steps...")
        for _ in range(args.warmup_steps):
            run_step()

    print("benchmarking...")
    step_times = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    if args.memory_snapshot:
        torch.cuda.memory._record_memory_history(max_entries=1_000_000)

    for _ in range(args.num_steps):
        start_time = timeit.default_timer()
        run_step()
        end_time = timeit.default_timer()
        step_times.append(end_time - start_time)

    if args.memory_snapshot:
        torch.cuda.memory._dump_snapshot(args.memory_snapshot)
        torch.cuda.memory._record_memory_history(enabled=None)
        print(f"saved memory snapshot to {args.memory_snapshot}")

    step_times = np.array(step_times)
    mean_time = np.mean(step_times)
    std_time = np.std(step_times)

    print(f"results ({args.mode}):")
    print(f"mean time: {mean_time * 1000:.2f} ms")
    print(f"std dev: {std_time * 1000:.2f} ms")
    if device.type == "cuda":
        print(f"peak memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")


if __name__ == "__main__":
    main()
