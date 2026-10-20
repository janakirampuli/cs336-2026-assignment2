# Assignment 2 log

## End-to-end benchmarking (2.1.3)

All runs use one H100, fp32, batch size 4 and context length 512.

### Model sizes (problem b)

5 warmup steps and 10 timed steps. Each cell is the mean and standard deviation of the step time.

| Size | forward | forward + backward | train step |
|---|---|---|---|
| small | 20.96 ± 0.16 ms | 62.63 ± 0.43 ms | 71.44 ± 0.30 ms |
| medium | 57.66 ± 0.32 ms | 170.61 ± 0.60 ms | 195.28 ± 0.47 ms |
| large | 127.71 ± 0.73 ms | 388.73 ± 0.41 ms | 439.17 ± 0.73 ms |
| xl | 344.75 ± 1.82 ms | 1,073.29 ± 1.64 ms | 1,236.83 ± 0.77 ms |
| 10B | out of memory | out of memory | out of memory |

The 10B model ran out of memory in all three modes, including forward only. The failed allocation was 144 MiB with 78.16 GiB already allocated by PyTorch.

### Warmup steps (problem c)

0, 1 and 2 warmup steps, 10 timed steps each. The 5 column is from problem (b). Each cell is the mean and standard deviation in ms.

| Size | Mode | 0 warmup | 1 warmup | 2 warmup | 5 warmup |
|---|---|---|---|---|---|
| small | forward | 52.63 ± 96.99 | 20.43 ± 0.06 | 20.41 ± 0.04 | 20.96 ± 0.16 |
| small | backward | 97.89 ± 107.17 | 62.25 ± 0.32 | 61.60 ± 0.11 | 62.63 ± 0.43 |
| small | train | 108.32 ± 114.89 | 69.70 ± 0.25 | 70.68 ± 0.21 | 71.44 ± 0.30 |
| medium | forward | 85.74 ± 88.91 | 56.05 ± 0.13 | 56.19 ± 0.43 | 57.66 ± 0.32 |
| medium | backward | 211.31 ± 123.23 | 169.42 ± 0.31 | 170.31 ± 0.35 | 170.61 ± 0.60 |
| medium | train | 254.46 ± 181.97 | 192.15 ± 0.16 | 193.49 ± 0.58 | 195.28 ± 0.47 |
| large | forward | 153.00 ± 78.53 | 125.33 ± 0.90 | 127.04 ± 0.48 | 127.71 ± 0.73 |
| large | backward | 427.27 ± 119.45 | 386.02 ± 0.65 | 388.48 ± 0.35 | 388.73 ± 0.41 |
| large | train | 503.85 ± 204.26 | 481.30 ± 130.32 | 438.55 ± 0.31 | 439.17 ± 0.73 |
| xl | forward | 378.44 ± 122.43 | 337.58 ± 0.23 | 335.35 ± 0.55 | 344.75 ± 1.82 |
| xl | backward | 1,116.90 ± 141.53 | 1,069.59 ± 1.90 | 1,068.11 ± 0.57 | 1,073.29 ± 1.64 |
| xl | train | 1,287.40 ± 155.31 | 1,239.86 ± 0.50 | 1,234.80 ± 0.22 | 1,236.83 ± 0.77 |
