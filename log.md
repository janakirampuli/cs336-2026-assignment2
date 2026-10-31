# Assignment 2 log

All runs use one H100 GPU with fp32 numbers, batch size 4, and vocab size 10,000.

## Timing the model (2.1.3)

### Step time by model size (problem b)

The backward pass takes about twice as long as the forward pass, and the optimizer step adds about 12% more. Each run used context length 512, 5 warmup steps, and 10 timed steps. We got the backward and optimizer times by subtracting one mode's time from the next.

| Size | Parameters | Forward | Backward | Optimizer | Train step |
|---|---|---|---|---|---|
| small | 129M | 21 ms | 42 ms | 9 ms | 71 ms |
| medium | 423M | 58 ms | 113 ms | 25 ms | 195 ms |
| large | 969M | 128 ms | 261 ms | 50 ms | 439 ms |
| xl | 3.4B | 345 ms | 729 ms | 164 ms | 1,237 ms |
| 10B | 12.8B | out of memory | | | |

Bigger models use the GPU better. The xl model is about 26 times bigger than the small model, but its train step is only about 17 times slower, because big matrix multiplies keep more of the GPU busy.

The times barely change from step to step once the GPU is warmed up. The spread was under 1% of the average in every run.

The 10B model does not fit even for a forward pass. Its weights alone take about 51 GB, and the values saved for the backward pass fill the rest of the GPU's 80 GB.

### Warmup (problem c)

The first step in a new program is slow, and one warmup step fixes most of it. Without warmup, one of the 10 timed steps takes about 250 to 650 ms longer than the others. The extra time is about the same for every model size, so it comes from one time setup work on the GPU and not from the model's math.

| Size | Mode | 0 warmup | 1 warmup | 5 warmup |
|---|---|---|---|---|
| small | forward | 53 ± 97 ms | 20 ms | 21 ms |
| small | train | 108 ± 115 ms | 70 ms | 71 ms |
| xl | forward | 378 ± 122 ms | 338 ms | 345 ms |
| xl | train | 1,287 ± 155 ms | 1,240 ms | 1,237 ms |

With one or two warmup steps, almost every run comes within about 3% of the 5 warmup result. The one exception was the large model's train step with one warmup step, where one step was still about 420 ms slow.

## Profiling the GPU (2.1.4)

We profiled train steps with Nsight Systems, a tool that records every function the GPU runs. Each of those GPU functions is called a kernel. We put named markers around the forward pass, backward pass, optimizer step, and parts of attention, so we could see which kernels belong to which part. The reports are in `runs/nsys/`.

### Memory limits by context length

Memory runs out fast as the context gets longer, because every layer stores an attention table with one number for each pair of tokens. Doubling the context makes that table 4 times bigger. The small model fits a context of 2048 but not 4096. The large model fits 1024 but not 2048, and the xl model does not fit 1024, so we profiled small and large.

### Where the GPU time goes

Matrix multiplies take most of the time, but not all of it. In the small model at context 512, they take about 70% of the forward pass and about 60% of a full train step. The rest goes to many small operations like add, multiply, divide, exp, and masking. Each of those reads every number from memory, does one or two math steps on it, and writes it back. The GPU can do math much faster than it can read and write memory, so these small operations spend most of their time waiting on memory.

| Small model, context 512 | Forward | Backward | Optimizer |
|---|---|---|---|
| GPU time per step | 19 ms | 39 ms | 6 ms |
| Share in matrix multiplies | 71% | 66% | 0% |
| Number of kernels | about 630 | about 1,280 | about 1,780 |

The backward pass takes about twice as long as the forward pass in every profile, the same as in the timing runs.

The optimizer step has no matrix multiplies. AdamW runs 16 small kernels for each weight tensor, which comes to about 1,800 kernels per step for the small model and about 5,200 for the large one. Its time depends only on the number of weights, so it stays the same at every context length, at about 6 ms for small and 44 ms for large.

### Attention and context length

Attention gets slower faster than the rest of the model as the context grows. Each time the context doubles, attention takes about 4 times longer and the rest of the forward pass takes about 2 times longer, because the attention table grows with the square of the context. The table below shows the small model.

| Context | Forward pass | Share in attention | Share in matrix multiplies |
|---|---|---|---|
| 256 | 9 ms | 13% | 77% |
| 512 | 19 ms | 24% | 71% |
| 1024 | 42 ms | 39% | 62% |
| 2048 | 113 ms | 55% | 52% |

The large model shows the same trend more slowly, going from about 10% in attention at context 256 to about 30% at 1024, because its other layers are wider and take more of the time.

Inside attention, the softmax is slower than the matrix multiply before it, even though it does much less math. The matrix multiply that makes the attention scores does about 128 math operations per score, and the softmax does about 5. Still, the softmax takes about 1.8 times as long, and the masking step alone takes about half as long as the matrix multiply. The matrix multiply reuses each number it reads many times. The softmax runs as 5 separate kernels, and each one reads and writes the whole attention table, so its time goes to memory traffic and not to math.

| Small model, context 2048 | GPU time per forward pass |
|---|---|
| Score matrix multiply and scaling | 16 ms |
| Masking | 8 ms |
| Softmax | 31 ms |
| Final matrix multiply | 7 ms |

### GPU waiting on Python

When the work is small, the GPU can sit idle waiting for Python to send the next kernel. For the small model at context 256, the forward pass had about 9 ms of GPU work spread over about 19 ms, so the GPU was idle about half the time. The optimizer step was worse, with about 6 ms of work spread over about 20 ms, because it sends many tiny kernels. Bigger models and longer contexts give each kernel more work, and the idle gaps mostly go away.

### Profile and timing agree

The profile matches the timing script. For the small model at context 512, the forward pass took about 21 ms on the GPU, and the timing script measured 21 ms. Profiling adds a little extra work, so train steps under the profiler took about 2% to 11% longer.

## Mixed precision (2.1.5)

### Number formats

A float stores a number like scientific notation, as some digits times a power of 2. The exponent bits decide how big or small a number can get, and the digit bits (the mantissa) decide how precise it is. fp16 has a small range and medium precision, while bf16 has the same range as fp32 but even less precision than fp16.

| | fp32 | fp16 | bf16 |
|---|---|---|---|
| Exponent bits | 8 | 5 | 8 |
| Digit bits (mantissa) | 23 | 10 | 7 |
| Largest number | about 3.4e38 | 65,504 | about 3.4e38 |
| Smallest normal number | about 1.2e-38 | about 0.00006 | about 1.2e-38 |
| Gap between 1 and the next number | about 0.0000001 | about 0.001 | about 0.008 |
| Significant digits | about 7 | about 3 | about 2 |

Between any two powers of two, like 4 and 8, the numbers a format can store are evenly spaced. fp16 fits 1,024 numbers in each of those stretches and bf16 fits only 128. Each stretch is twice as wide as the one before it, so the gap between stored numbers doubles at 1, 2, 4, 8, and so on.

### Adding small numbers (mixed_precision_accumulation)

Adding up many small numbers in low precision goes wrong, but storing the numbers themselves in low precision is fine. Each run adds 0.01 a thousand times, so the right answer is 10.

| Running total | Numbers added | Result | Error |
|---|---|---|---|
| fp32 | fp32 | 10.0001 | about 0.001% |
| fp16 | fp16 | 9.953 | about 0.5% |
| fp32 | fp16 | 10.002 | about 0.02% |
| fp32 | fp16 turned into fp32 | 10.002 | about 0.02% |
| bf16 | bf16 | 4.0 | 60% |
| fp32 | bf16 | 10.010 | about 0.1% |

An fp32 total of fp16 numbers is only a little off, because fp16 stores 0.01 as 0.0100021, and 1000 of those add up to 10.0021. That error is small, and it does not grow as the total grows. The fourth row gives the same answer as the third because PyTorch already turns the fp16 number into fp32 before adding it to an fp32 total.

An fp16 total is worse, because every add gets rounded to the nearest number fp16 can store. Between 4 and 8, stored numbers are about 0.0039 apart, so each 0.01 rounds up to 0.0117 and the total runs ahead. Above 8 they are about 0.0078 apart, so each 0.01 rounds down to 0.0078, and the total falls behind and ends 0.05 short.

A bf16 total stops growing. Above 4, stored bf16 numbers are 0.03125 apart, and 0.01 is less than half of that gap, so each add rounds back to the same number. The total got stuck at exactly 4 after 350 adds. Storing the numbers in bf16 is still fine, because an fp32 total of bf16 numbers lands at 10.010.

This is why mixed precision keeps running totals, like sums and averages, in fp32 even when the numbers being added are fp16 or bf16. Matrix multiply kernels do the same, because they add up each row times column in fp32.

### Number types under fp16 autocast (benchmarking_mixed_precision a)

Autocast changes the number type of each operation's output, but it does not change the stored weights. We ran the handout's ToyModel on an L4 GPU with fp32 weights inside `torch.autocast(dtype=torch.float16)`.

| Point in the model | Number type |
|---|---|
| Weights inside autocast | fp32 |
| Output of fc1 | fp16 |
| Output of layer norm | fp32 |
| Logits (output of fc2) | fp16 |
| Loss | fp32 |
| Gradients | fp32 |

The linear layers run in fp16 and make an fp16 copy of their fp32 weights for each call. Layer norm runs in fp32 even though its input is fp16, and fc2 turns the result back into fp16. The loss runs in fp32, and the gradients are fp32 because they belong to the fp32 weights.

### bf16 speed by model size (benchmarking_mixed_precision c)

bf16 makes the forward and backward passes faster, and the gain gets bigger as the model gets bigger. The small model gets about 1.5 to 2 times faster and xl gets about 4.5 times faster. The settings match problem (b).

| Size | Forward, fp32 to bf16 | Backward, fp32 to bf16 | Optimizer, fp32 to bf16 | Train step, fp32 to bf16 |
|---|---|---|---|---|
| small | 21 to 14 ms (1.5x) | 42 to 22 ms (1.9x) | 9 to 10 ms | 71 to 46 ms (1.6x) |
| medium | 58 to 28 ms (2.1x) | 113 to 49 ms (2.3x) | 25 to 25 ms | 195 to 101 ms (1.9x) |
| large | 128 to 45 ms (2.8x) | 261 to 94 ms (2.8x) | 50 to 53 ms | 439 to 192 ms (2.3x) |
| xl | 345 to 76 ms (4.6x) | 729 to 171 ms (4.3x) | 164 to 164 ms | 1,237 to 411 ms (3.0x) |

The H100 has special hardware for bf16 matrix multiplies, called tensor cores, which is much faster than its fp32 hardware. Bigger models spend more of their time in matrix multiplies, so they gain more. The small model gains less, because more of its time goes to small operations and to waiting on Python, and autocast adds extra kernels that convert numbers to bf16.

The optimizer step does not get faster at all, because autocast only covers the forward pass, and AdamW still updates fp32 weights. For xl, the optimizer goes from about 13% of the train step in fp32 to about 40% in bf16, so it becomes the next thing worth speeding up.

The 10B model still runs out of memory in the forward pass, at about 78 GiB used. The fp32 weights take about 48 GiB, and autocast keeps a bf16 copy of each weight during the forward pass, which adds about 24 GiB more.

## Memory profiling (2.1.6)

We used the small model instead of xl, because xl does not fit a context of 2048. Each run recorded 2 steps after 2 warmup steps. The snapshots are in `runs/memory/` and open at pytorch.org/memory_viz.

### Peak memory (problems b and c)

GPU memory holds two kinds of things. The first kind stays the same size no matter the context length, and it is the weights plus, for training, the gradients and the two AdamW state tensors. The second kind is the activations, which are the values the forward pass saves for the backward pass, and they grow with the context length.

| Context | Mode | fp32 peak | bf16 peak |
|---|---|---|---|
| 128 | forward | 1.2 GiB | 1.2 GiB |
| 128 | train | 2.2 GiB | 2.3 GiB |
| 2048 | forward | 28 GiB | 20 GiB |
| 2048 | train | 31 GiB | 24 GiB |

At context 128, the fixed part takes most of the memory. The small model's weights take about 0.5 GiB in fp32, and weights, gradients, and the two AdamW states together take about 2 GiB, which is most of the 2.2 GiB train peak.

At context 2048, the activations take almost all of the memory. Going from 128 to 2048 tokens is 16 times more tokens, but the forward peak grows about 24 times, because each layer's attention table grows with the square of the context.

bf16 only saves memory when activations take most of the memory, and it saves less than half. At context 128 it saves nothing, because the weights, gradients, and AdamW states stay fp32. At context 2048 it saves about a quarter. Autocast likely still keeps some activations in fp32, like the attention tables inside the softmax, since exp and sum run in fp32 under autocast.

The forward runs keep autograd on, so their peaks include the values saved for a backward pass that never runs. A forward pass inside `torch.no_grad()` would need much less memory.

## Gradient checkpointing (3)

### What one block saves

One block of our model saves 10 tensors the size of the hidden state, 5 tensors the size of the feed forward layer, and 2 attention tables for its backward pass. We counted them with the saved tensor hooks from the handout. For xl at batch 4 and context 2048 in fp32, that is about 6.3 GiB per block, and the 2 attention tables alone are 4 GiB of it. The block's input, which is all a checkpoint keeps, is only 80 MiB.

### Recursive checkpointing on a toy model (problem a)

Splitting the blocks in half again and again keeps the gradients exactly the same as a plain run, and each block reruns its forward pass once per level of splitting. With 8 small blocks and early stopping turned off, every block ran its forward pass 4 times, which is log₂(8) + 1. With PyTorch's default early stopping, the counts were 4, 3, 3, 2 in each half, because a recomputed group stops as soon as it has rebuilt the inputs of its inner checkpoints.

### Segment size on xl (problem b)

Checkpointing every single block gives the lowest peak, because a checkpoint costs about 80 times less than a block of residuals. We ran xl at context 2048 and batch 4 in fp32, with forward and backward but no optimizer step, and put every k layers inside one checkpoint.

| Layers per checkpoint | Checkpoints | Predicted activations | Measured peak | Step time |
|---|---|---|---|---|
| 1 | 32 | 9 GiB | 38 GiB | 6.4 s |
| 2 | 16 | 14 GiB | 44 GiB | 6.5 s |
| 4 | 8 | 26 GiB | 56 GiB | 6.6 s |
| 8 | 4 | 51 GiB | out of memory | |
| no checkpointing | 0 | 203 GiB | out of memory | |

The prediction is the stored checkpoints plus one recomputed group of k blocks, which is (32 / k) × 80 MiB + k × 6.3 GiB. Setting both parts equal gives a best group size of about 0.6 blocks, so 1 is the best size we can use. The measured gaps match the prediction well. Going from 1 to 2 layers per checkpoint added 6 GiB against a predicted 5 GiB, and going from 1 to 4 added 18 GiB against a predicted 17 GiB.

The measured peaks sit about 29 GiB above the predicted activations. Most of that is the weights and gradients, which take about 25 GiB for 3.4B parameters in fp32. The rest is the logits and loss buffers, plus attention tables that exist for a short time while a block is recomputed.

The step time barely changes with the group size, because every block reruns its forward pass exactly once no matter how the blocks are grouped. Without checkpointing, xl does not fit at context 2048 even without the AdamW state, so checkpointing is the only way to run it here.
