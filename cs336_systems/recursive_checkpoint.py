import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def run_blocks(blocks: list[nn.Module], x: torch.Tensor) -> torch.Tensor:
    if len(blocks) == 0:
        return x
    if len(blocks) == 1:
        return blocks[0](x)

    mid = len(blocks) // 2
    x = checkpoint(run_blocks, blocks[:mid], x, use_reentrant=False)
    x = checkpoint(run_blocks, blocks[mid:], x, use_reentrant=False)
    return x


class CountedBlock(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.linear = nn.Linear(d, d)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return torch.tanh(self.linear(x))


def check(num_blocks: int = 8, d: int = 16):
    torch.manual_seed(0)
    blocks = [CountedBlock(d) for _ in range(num_blocks)]
    x = torch.randn(4, d, requires_grad=True)

    y = x
    for block in blocks:
        y = block(y)
    y.sum().backward()
    expected = [p.grad.clone() for b in blocks for p in b.parameters()]

    for b in blocks:
        b.calls = 0
        b.zero_grad()
    run_blocks(blocks, x).sum().backward()
    actual = [p.grad for b in blocks for p in b.parameters()]

    print(f"N={num_blocks}")
    print("grads match:", all(torch.allclose(e, a) for e, a in zip(expected, actual)))
    print("forward calls per block:", [b.calls for b in blocks])


if __name__ == "__main__":
    check(8)
    check(16)
