import torch

s = torch.tensor(0, dtype=torch.float32)
for _ in range(1000):
    s += torch.tensor(0.01, dtype=torch.float32)
print(f"fp32 sum, fp32 values: {s.item():.6f}")

s = torch.tensor(0, dtype=torch.float16)
for _ in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)
print(f"fp16 sum, fp16 values: {s.item():.6f}")

s = torch.tensor(0, dtype=torch.float32)
for _ in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)
print(f"fp32 sum, fp16 values: {s.item():.6f}")

s = torch.tensor(0, dtype=torch.float32)
for _ in range(1000):
    x = torch.tensor(0.01, dtype=torch.float16)
    s += x.type(torch.float32)
print(f"fp32 sum, fp16 values cast to fp32: {s.item():.6f}")

s = torch.tensor(0, dtype=torch.bfloat16)
for _ in range(1000):
    s += torch.tensor(0.01, dtype=torch.bfloat16)
print(f"bf16 sum, bf16 values: {s.item():.6f}")

s = torch.tensor(0, dtype=torch.float32)
for _ in range(1000):
    s += torch.tensor(0.01, dtype=torch.bfloat16)
print(f"fp32 sum, bf16 values: {s.item():.6f}")

print(f"0.01 stored in fp32: {torch.tensor(0.01, dtype=torch.float32).item():.10f}")
print(f"0.01 stored in fp16: {torch.tensor(0.01, dtype=torch.float16).item():.10f}")
print(f"0.01 stored in bf16: {torch.tensor(0.01, dtype=torch.bfloat16).item():.10f}")
