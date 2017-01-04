# CUDA_Blocked_APSP
CUDA APSP using Blocked Floyd-Warshall

## Usage

```shell
make
./apsp N [1]

# N: the size of matrix. N = 2^k
# 1: 1 means validate; ignore if not provided
```

## Example

```shell
-> % ./apsp 512 1
CUDA_APSP elasped time = 165.30 us
CUDA_APSP elasped time = 75.80 us
CUDA_APSP elasped time = 75.50 us
CUDA_APSP elasped time = 75.50 us
CUDA_APSP elasped time = 75.50 us
CUDA_APSP elasped time = 75.30 us
CUDA_APSP elasped time = 75.70 us
CUDA_APSP elasped time = 75.60 us
CUDA_APSP elasped time = 75.50 us
CUDA_APSP elasped time = 75.70 us
Your result is correct.
ST_APSP elasped time   = 1150693.00 us
============================================
Speedup = 1520.07
```


```shell
-> % ./apsp 4096
CUDA_APSP elasped time = 24764.90 us
CUDA_APSP elasped time = 17498.10 us
CUDA_APSP elasped time = 16643.40 us
CUDA_APSP elasped time = 16643.70 us
CUDA_APSP elasped time = 16643.10 us
CUDA_APSP elasped time = 16644.50 us
CUDA_APSP elasped time = 16643.80 us
CUDA_APSP elasped time = 16642.50 us
CUDA_APSP elasped time = 16644.80 us
CUDA_APSP elasped time = 16641.00 us
```
