# Assignment 2 log

All runs use one H100 GPU with fp32 numbers, batch size 4, and vocab size 10,000.

## Timing the model (2.1.3)

### Step time by model size (problem b)

The backward pass takes about twice as long as the forward pass, and the optimizer step adds about 12% more. Each run used context length 512, with 5 warmup steps and 10 timed steps. We got the backward and optimizer times by subtracting one mode's time from the next.

| Size | Parameters | Forward | Backward | Optimizer | Train step |
|---|---|---|---|---|---|
| small | 129M | 21 ms | 42 ms | 9 ms | 71 ms |
| medium | 423M | 58 ms | 113 ms | 25 ms | 195 ms |
| large | 969M | 128 ms | 261 ms | 50 ms | 439 ms |
| xl | 3.4B | 345 ms | 729 ms | 164 ms | 1,237 ms |
| 10B | 12.8B | out of memory | | | |

Bigger models keep more of the GPU busy, because their matrix multiplies are bigger. For example, the xl model is about 26 times bigger than the small model, but its train step is only about 17 times slower.

The times barely change from step to step once the GPU is warmed up, and the spread was under 1% of the average in every run.

The 10B model does not fit even for a forward pass. Its weights alone take about 51 GB, and the values saved for the backward pass fill the rest of the GPU's 80 GB.

### Warmup (problem c)

The first step in a new program is slow, and one warmup step fixes most of it. Without warmup, one of the 10 timed steps takes about 250 to 650 ms longer than the others. The extra time is about the same for every model size, so it comes from setup work that the GPU does once, and not from the model's math.

| Size | Mode | 0 warmup | 1 warmup | 5 warmup |
|---|---|---|---|---|
| small | forward | 53 ± 97 ms | 20 ms | 21 ms |
| small | train | 108 ± 115 ms | 70 ms | 71 ms |
| xl | forward | 378 ± 122 ms | 338 ms | 345 ms |
| xl | train | 1,287 ± 155 ms | 1,240 ms | 1,237 ms |

With one or two warmup steps, almost every run comes within about 3% of the result with 5 warmup steps. The one exception was the large model's train step with one warmup step, where one step was still about 420 ms slow.

## Profiling the GPU (2.1.4)

We profiled train steps with Nsight Systems, a tool that records every function the GPU runs, and each of those GPU functions is called a kernel. We put named markers around the forward pass, the backward pass, the optimizer step, and parts of attention, so we could see which kernels belong to which part. The reports are in `runs/nsys/`.

### Memory limits by context length

Memory runs out fast as the context gets longer, because every layer stores an attention table with one number for each pair of tokens. Doubling the context makes the table 4 times bigger. The small model fits a context of 2048 but not 4096, and the large model fits 1024 but not 2048. The xl model does not fit 1024, so we profiled small and large.

### Where the GPU time goes

Matrix multiplies take most of the time, but not all of it. In the small model at context 512, they take about 70% of the forward pass and about 60% of a full train step. The rest goes to many small operations, e.g., add, multiply, divide, exp, and masking. Each small operation reads every number from memory and does one or two math steps on it, and then it writes the result back. The GPU can do math much faster than it can read and write memory, so the small operations spend most of their time waiting on memory.

| Small model, context 512 | Forward | Backward | Optimizer |
|---|---|---|---|
| GPU time per step | 19 ms | 39 ms | 6 ms |
| Share in matrix multiplies | 71% | 66% | 0% |
| Number of kernels | about 630 | about 1,280 | about 1,780 |

The backward pass takes about twice as long as the forward pass in every profile, the same as in the timing runs.

The optimizer step has no matrix multiplies. AdamW runs 16 small kernels for each weight tensor, which comes to about 1,800 kernels per step for the small model and about 5,200 for the large one. Its time depends only on the number of weights, so it stays the same at every context length, at about 6 ms for small and 44 ms for large.

### Attention and context length

Attention gets slower faster than the rest of the model as the context grows, because the attention table grows with the square of the context. Each time the context doubles, attention takes about 4 times longer and the rest of the forward pass takes about 2 times longer. The table below shows the small model.

| Context | Forward pass | Share in attention | Share in matrix multiplies |
|---|---|---|---|
| 256 | 9 ms | 13% | 77% |
| 512 | 19 ms | 24% | 71% |
| 1024 | 42 ms | 39% | 62% |
| 2048 | 113 ms | 55% | 52% |

The large model shows the same trend more slowly, going from about 10% in attention at context 256 to about 30% at 1024. Its other layers are wider, so they take more of the time.

Inside attention, the softmax is slower than the matrix multiply before it, even though it does much less math. The matrix multiply that makes the attention scores does about 128 math operations per score, and the softmax does about 5. Still, the softmax takes about 1.8 times as long, and the masking step alone takes about half as long as the matrix multiply. The matrix multiply reuses each number it reads many times. The softmax runs as 5 separate kernels that each read and write the whole attention table, so its time goes to memory traffic and not to math.

| Small model, context 2048 | GPU time per forward pass |
|---|---|
| Score matrix multiply and scaling | 16 ms |
| Masking | 8 ms |
| Softmax | 31 ms |
| Final matrix multiply | 7 ms |

### GPU waiting on Python

When the work is small, the GPU can sit idle waiting for Python to send the next kernel. For example, the small model's forward pass at context 256 had about 9 ms of GPU work spread over about 19 ms, so the GPU was idle about half the time. The optimizer step was worse, with about 6 ms of work spread over about 20 ms, because it sends many tiny kernels. Bigger models and longer contexts give each kernel more work, and the idle gaps mostly go away.

### Profile and timing agree

The profile matches the timing script. For the small model at context 512, the forward pass took about 21 ms on the GPU, and the timing script measured 21 ms. Profiling adds a little extra work, so train steps under the profiler took about 2% to 11% longer.

## Mixed precision (2.1.5)

### Number formats

A float stores a number as some digits times a power of 2. The exponent bits decide how big or small a number can get, and the digit bits (the mantissa) decide how precise it is. fp16 has a small range and medium precision, while bf16 has the same range as fp32 but even less precision than fp16.

| | fp32 | fp16 | bf16 |
|---|---|---|---|
| Exponent bits | 8 | 5 | 8 |
| Digit bits (mantissa) | 23 | 10 | 7 |
| Largest number | about 3.4e38 | 65,504 | about 3.4e38 |
| Smallest normal number | about 1.2e-38 | about 0.00006 | about 1.2e-38 |
| Gap between 1 and the next number | about 0.0000001 | about 0.001 | about 0.008 |
| Significant digits | about 7 | about 3 | about 2 |

Between any two powers of two, e.g., 4 and 8, the numbers a format can store are evenly spaced. fp16 fits 1,024 numbers in each of those stretches, and bf16 fits only 128. Each stretch is twice as wide as the one before it, so the gap between stored numbers doubles at 1, 2, 4, 8, and so on.

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

An fp32 total of fp16 numbers is only a little off, because fp16 stores 0.01 as 0.0100021. A thousand copies of 0.0100021 add up to 10.0021, and the error stays the same size as the total grows. The fourth row gives the same answer as the third, because PyTorch already turns the fp16 number into fp32 before adding it to an fp32 total.

An fp16 total is worse, because every add gets rounded to the nearest number fp16 can store. Between 4 and 8, stored numbers are about 0.0039 apart, so each 0.01 rounds up to 0.0117 and the total runs ahead. Above 8 they are about 0.0078 apart, so each 0.01 rounds down to 0.0078. The total then falls behind, and it ends 0.05 short.

A bf16 total stops growing. Above 4, stored bf16 numbers are 0.03125 apart, and 0.01 is less than half of the gap. Each add therefore rounds back to the same number, and the total got stuck at exactly 4 after 350 adds. Storing the numbers in bf16 is still fine, because an fp32 total of bf16 numbers lands at 10.010.

Mixed precision keeps running totals, e.g., sums and averages, in fp32 even when the numbers being added are fp16 or bf16, so the rounding problem above doesn't happen. Matrix multiply kernels do the same, because they add up each row times column in fp32.

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

bf16 makes the forward and backward passes faster, and the gain gets bigger as the model gets bigger. The small model gets about 1.5 to 2 times faster, and xl gets about 4.5 times faster. The settings match problem (b).

| Size | Forward, fp32 to bf16 | Backward, fp32 to bf16 | Optimizer, fp32 to bf16 | Train step, fp32 to bf16 |
|---|---|---|---|---|
| small | 21 to 14 ms (1.5x) | 42 to 22 ms (1.9x) | 9 to 10 ms | 71 to 46 ms (1.6x) |
| medium | 58 to 28 ms (2.1x) | 113 to 49 ms (2.3x) | 25 to 25 ms | 195 to 101 ms (1.9x) |
| large | 128 to 45 ms (2.8x) | 261 to 94 ms (2.8x) | 50 to 53 ms | 439 to 192 ms (2.3x) |
| xl | 345 to 76 ms (4.6x) | 729 to 171 ms (4.3x) | 164 to 164 ms | 1,237 to 411 ms (3.0x) |

The H100 has separate hardware for bf16 matrix multiplies, called tensor cores, and it is much faster than the fp32 hardware. Bigger models spend more of their time in matrix multiplies, so they gain more. The small model gains less, because more of its time goes to small operations and to waiting on Python. Autocast also adds extra kernels that convert numbers to bf16.

The optimizer step does not get faster at all, because autocast only covers the forward pass, and AdamW still updates fp32 weights. For xl, the optimizer goes from about 13% of the train step in fp32 to about 40% in bf16, so after bf16 the optimizer is the largest part left to speed up.

The 10B model still runs out of memory in the forward pass, at about 78 GiB used. The fp32 weights take about 48 GiB, and autocast keeps a bf16 copy of each weight during the forward pass, which adds about 24 GiB more.

## Memory profiling (2.1.6)

We used the small model instead of xl, because xl does not fit a context of 2048. Each run recorded 2 steps after 2 warmup steps. The snapshots are in `runs/memory/` and open at pytorch.org/memory_viz.

### Peak memory (problems b and c)

GPU memory holds two kinds of things. The first kind stays the same size at any context length, and it is the weights plus, for training, the gradients and the two AdamW state tensors. The second kind is the activations, which are the values the forward pass saves for the backward pass, and they grow with the context length.

| Context | Mode | fp32 peak | bf16 peak |
|---|---|---|---|
| 128 | forward | 1.2 GiB | 1.2 GiB |
| 128 | train | 2.2 GiB | 2.3 GiB |
| 2048 | forward | 28 GiB | 20 GiB |
| 2048 | train | 31 GiB | 24 GiB |

At context 128, the fixed part takes most of the memory. The small model's weights take about 0.5 GiB in fp32, and the weights, gradients, and two AdamW states together take about 2 GiB, which is most of the 2.2 GiB train peak.

At context 2048, the activations take almost all of the memory. Going from 128 to 2048 tokens gives 16 times more tokens, but the forward peak grows about 24 times, because each layer's attention table grows with the square of the context.

bf16 saves memory only when activations take most of the memory, and even then it saves less than half. At context 128 it saves nothing, because the weights, gradients, and AdamW states stay fp32. At context 2048 it saves about a quarter. Autocast likely still keeps some activations in fp32, e.g., the attention tables inside the softmax, since exp and sum run in fp32 under autocast.

The forward runs keep autograd on, so their peaks include the values saved for a backward pass that never runs. A forward pass inside `torch.no_grad()` would need much less memory.

## Gradient checkpointing (3)

### What one block saves

One block of our model saves 10 tensors the size of the hidden state, 5 tensors the size of the feed forward layer, and 2 attention tables for its backward pass. We counted them with the saved tensor hooks from the handout. For xl at batch 4 and context 2048 in fp32, the saved tensors take about 6.3 GiB per block, and the 2 attention tables alone are 4 GiB of the total. A checkpoint keeps only the block's input, which is 80 MiB.

### Recursive checkpointing on a toy model (problem a)

Splitting the blocks in half again and again keeps the gradients exactly the same as a plain run, and each block reruns its forward pass once per level of splitting. With 8 small blocks and early stopping turned off, every block ran its forward pass 4 times, which is log₂(8) + 1. With PyTorch's default early stopping, the counts were 4, 3, 3, 2 in each half. A recomputed group stops as soon as it has rebuilt the inputs of its inner checkpoints, so the later blocks in each half rerun fewer times.

### Segment size on xl (problem b)

Checkpointing every single block gives the lowest peak, because a checkpoint takes about 80 times less memory than the tensors one block saves for its backward pass. We ran xl at context 2048 and batch 4 in fp32, with forward and backward but no optimizer step, and put every k layers inside one checkpoint.

| Layers per checkpoint | Checkpoints | Predicted activations | Measured peak | Step time |
|---|---|---|---|---|
| 1 | 32 | 9 GiB | 38 GiB | 6.4 s |
| 2 | 16 | 14 GiB | 44 GiB | 6.5 s |
| 4 | 8 | 26 GiB | 56 GiB | 6.6 s |
| 8 | 4 | 51 GiB | out of memory | |
| no checkpointing | 0 | 203 GiB | out of memory | |

The prediction adds the stored checkpoints to the saved tensors of one recomputed group of k blocks, which is (32 / k) × 80 MiB + k × 6.3 GiB. Setting both parts equal gives a best group size of about 0.6 blocks, so 1 is the best size we can use. The measured gaps match the prediction well. Going from 1 to 2 layers per checkpoint added 6 GiB against a predicted 5 GiB, and going from 1 to 4 added 18 GiB against a predicted 17 GiB.

The measured peaks sit about 29 GiB above the predicted activations. Most of the gap is the weights and gradients, which take about 25 GiB for 3.4B parameters in fp32. The rest is the logits and loss buffers, plus attention tables that exist for a short time while a block is recomputed.

The step time barely changes with the group size, because every block reruns its forward pass exactly once however the blocks are grouped. Without checkpointing, xl does not fit at context 2048 even without the AdamW state, so checkpointing is the only way to run it here.

## Attention on its own (4.1.1)

### Time and memory by sequence length and head size (pytorch_attention a)

The attention tables decide both the time and the memory, so the sequence length matters and the head size barely does. We timed the staff attention function alone in fp32 with batch 8 and no head dimension, averaging 100 forward and 100 backward passes on the H100.

| Sequence length | Forward (d = 16 to 128) | Backward (d = 16 to 128) | Memory before backward |
|---|---|---|---|
| 256 | 0.11 ms | 0.35 to 0.38 ms | 0.07 GiB |
| 1,024 | 0.22 to 0.29 ms | 0.62 to 0.72 ms | 0.13 to 0.14 GiB |
| 4,096 | 2.6 to 3.5 ms | 6.2 to 8.1 ms | 1.1 GiB |
| 8,192 | 10 to 14 ms | 24 to 31 ms | 4.1 to 4.2 GiB |
| 16,384 | 39 to 55 ms | 95 to 124 ms | 16.1 to 16.4 GiB |

Nothing ran out of memory on the 80 GB H100, even at 16,384 tokens.

The memory before the backward pass is the two saved attention tables. One table at batch 8 and 16,384 tokens is 8 × 16,384 × 16,384 × 4 bytes = 8 GiB. Two tables make 16 GiB, which matches the measurement to within 0.4 GiB, and Q, K, V, and the output take only 8 to 64 MiB each. Every time the sequence length doubles, the memory grows about 4 times, because each table grows with the square of the sequence length.

The time also grows about 4 times per doubling once the sequence is long, for the same reason. At 256 and 1,024 tokens the work is so small that the time is mostly the cost of launching kernels. Going from d = 16 to d = 128 adds only about 40% at 16,384 tokens, even though Q, K, and V get 8 times bigger, because most of the time goes to reading and writing the tables in the softmax and the masking.

The backward pass takes about 2.4 times as long as the forward pass, since it has two matrix multiplies for each one in the forward pass and makes several more passes over the tables.

### Where it would run out of memory (worked out, not run)

The first size that should run out of memory on the 80 GB H100 is 32,768 tokens, and it should fail in the forward pass, before anything is saved for the backward pass. One table at 32,768 tokens is 8 × 32,768 × 32,768 × 4 bytes = 32 GiB, which is 4 times the 8 GiB at 16,384 tokens.

The forward pass keeps several tables alive at the same moment. Inside the softmax, the scaled scores, the scores minus the row max, and the result of exp all exist together, which is 3 tables or 96 GiB, and 96 GiB is already more than the GPU holds. At 16,384 tokens the same moment needs 3 × 8 = 24 GiB. The largest moment there comes when the division by the row sum makes a fourth table, and it needs 32 GiB, so it fits. Even if the forward pass got through at 32,768 tokens, the two saved tables would take 64 GiB before the backward pass starts.

### Compiled attention (torch_compile a)

`torch.compile` makes attention 2 to 3 times faster at long sequences but saves no memory. We ran the same sweep with `torch.compile(scaled_dot_product_attention)`, compiling fresh for each shape and leaving compile time out of the timings. The table shows the uncompiled time, then the compiled time, for the smallest and largest head sizes.

| Sequence length | Forward, d = 16 | Forward, d = 128 | Backward, d = 16 | Backward, d = 128 |
|---|---|---|---|---|
| 256 | 0.15 to 0.15 ms | 0.15 to 0.14 ms | 0.45 to 0.36 ms | 0.49 to 0.34 ms |
| 1,024 | 0.23 to 0.15 ms | 0.30 to 0.22 ms | 0.64 to 0.41 ms | 0.75 to 0.53 ms |
| 4,096 | 2.6 to 1.0 ms (2.7x) | 3.6 to 1.9 ms (1.8x) | 6.2 to 2.6 ms (2.4x) | 8.1 to 4.4 ms (1.8x) |
| 8,192 | 10 to 3.4 ms (2.9x) | 14 to 7.3 ms (1.9x) | 24 to 9.8 ms (2.5x) | 31 to 17 ms (1.8x) |
| 16,384 | 39 to 14 ms (2.9x) | 55 to 30 ms (1.9x) | 95 to 38 ms (2.5x) | 124 to 67 ms (1.8x) |

The compiled version joins the scaling, the softmax steps, and their backward steps into a few kernels, so it reads and writes each attention table fewer times. The passes over the tables are most of the time, so the speedup is biggest at long sequences. The speedup is also bigger for d = 16 than for d = 128. With a small head size almost all the time goes to the passes over the tables, while with a large head size more of the time goes to the matrix multiplies, which compiling does not speed up.

The memory before the backward pass is the same with and without compiling, about 16.5 GiB at 16,384 tokens. The compiled version still keeps two full attention tables for the backward pass, so it would run out of memory at the same sequence length. Getting rid of the tables needs a kernel that never writes them out in full, which is what FlashAttention does.

### Compiled model (torch_compile b)

Compiling the whole model helps much less than compiling attention alone, and it helps the small model most. We ran each size with and without `torch.compile(model)` at context 512 in fp32, with the same settings as the timing runs in 2.1.3. The table shows the uncompiled time, then the compiled time.

| Size | Forward | Forward and backward | Train step |
|---|---|---|---|
| small | 22 to 15 ms (1.4x) | 64 to 47 ms (1.35x) | 73 to 57 ms (1.3x) |
| medium | 56 to 44 ms (1.3x) | 174 to 134 ms (1.3x) | 197 to 157 ms (1.25x) |
| large | 125 to 104 ms (1.2x) | 391 to 322 ms (1.2x) | 440 to 372 ms (1.2x) |
| xl | 339 to 306 ms (1.1x) | 1,078 to 979 ms (1.1x) | 1,247 to 1,143 ms (1.1x) |

Compiling joins the small operations, e.g., the norms, the softmax, SiLU, RoPE, and the residual adds, into fewer kernels, but it leaves the matrix multiplies as they are. At context 512 the attention tables are small, so most of the time already goes to matrix multiplies. The gain is therefore much smaller than the 2 to 3 times we saw for attention at long sequences. Bigger models spend an even larger share of their time in matrix multiplies, so the gain falls from about 1.4 times for small to about 1.1 times for xl.

The optimizer step takes the same time with and without compiling, about 10 ms for small and 165 ms for xl, because only the model is compiled and AdamW still runs as many small separate operations.
