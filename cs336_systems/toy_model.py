import torch
from torch import nn


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        print(f"input: {x.dtype}")
        x = self.fc1(x)
        print(f"after fc1: {x.dtype}")
        x = self.relu(x)
        print(f"after relu: {x.dtype}")
        x = self.ln(x)
        print(f"after layernorm: {x.dtype}")
        x = self.fc2(x)
        print(f"after fc2: {x.dtype}")
        return x


def main():
    device = torch.device("cuda")
    model = ToyModel(in_features=32, out_features=5).to(device)
    inputs = torch.randn(4, 32, device=device)
    targets = torch.randint(0, 5, (4,), device=device)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(device.type)

    with torch.autocast(device_type=device.type, dtype=torch.float16):
        print(f"fc1 weight inside autocast: {model.fc1.weight.dtype}")
        logits = model(inputs)
        print(f"logits: {logits.dtype}")
        loss = loss_fn(logits, targets)
        print(f"loss: {loss.dtype}")

    scaler.scale(loss).backward()

    for name, param in model.named_parameters():
        print(f"{name} grad: {param.grad.dtype}")


if __name__ == "__main__":
    main()
