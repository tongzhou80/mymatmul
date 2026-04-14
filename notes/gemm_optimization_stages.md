This repo shows a few stages in optimizing GEMM or matrix multiplication (matmul). 
We do not distinguish GEMM from matmul in this documentation as their computations are basically the same except that GEMM adds a scaling factor and an offset when writing back 
the results. The underlying optimization principles are identical. Thus, matrix muliplication is used for illustration in this work for simplicity.
 
# Terminology
Consider matrix muliplication (matmul) as 

C = A x B

We note the dimensionality of the matrices as:

A: M x K
B: K x N
C: M x N

Sometimes the M, K, N dimensions are referred to as i, k, j as well, if the context is more code-oriented. A naive implementation of matmul is often expressed as 

```python
for i in range(M):
    for j in range(N):
        C[i,j] = 0
        for k in range(K):
            C[i,j] += A[i,k] * B[k,j] 
```

We will start with a naive "outer i-j parallelized" version where we launch 2-D grid with 2-D thread layout, and each thread loops over the entire k dimension to compute one output element. In CUDA style this would be
```
int i = blockDim.x * blockIdx.x + threadIdx.x;
int j = blockDim.y * blockIdx.y + threadIdx.y;

if (i < M && j < N) {
  float acc = 0f;
  for (int k = 0; k < K; k++) {
    acc += ....;
  }
  C[i*N + j] = acc;
}
``` 

Whatever thread layout is fine to start with, but let's use a 16x16 thread block layout for now.

From here, we will progress the optimizations in 4 stages:

# Stage 1: Memory Coalescing
The first optimization at this stage is to map the x-dim (thread layout dimension) to `j` instead of `i`, as `x-dim` is the fastest changing dimension in CUDA (a common
confusing design to CUDA beginners!) so it should be mapped to the `j` dimension for more contiguous memory access patterns.

```c
int j = blockDim.x * blockIdx.x + threadIdx.x;
int i = blockDim.y * blockIdx.y + threadIdx.y;
```

The next optimization in this phase is to make the thread layout x-dim a multiple of 32, the size of a warp, and the unit of thread execution in NV GPUs. This part does
not touch the CUDA kernel, and just changes the thread block layout from 16x16 to 32x8. Or 32x4 for a smaller thread block. The size of a thread block determines the GPU
occupancy but for this current implementation which is very simple, either 32x4 or 32x8 should have a similar occupancy.

This is known as "global memory coalescecing" in GPU terminology, where basically threads from the same warp access contiguous memory locations which can best utilize the 
global memory bandwidth.

# Stage 2: Shared Memory Tiling - Tradeoff Between Arith. Intensity and Occupancy
The kernels in stage 1 are easy to understand but they do not exploit any data reuse. Matmul itself actually has plenty of data reuse opportunity - at a $O(N^3)$ complexity
it loads only $O(N^2)$ data elements (in case of a square matrix). Being able to reuse data inside either shared memory or registers would significantly reduce the number
of global memory accesses, which is tens of thousands of times slower than the arithemic units (computes).

The simplest form of shared memory tiling is to still make each thread compute just one single output element, but reorganize the computation into 3 distinct phases:

- Phase 1: Threads cooperatively load a tile of A and a tile of B from global memory to on-chip shared memory.
- Phase 2: Each thread loops over a tile of the K dimension, and accumulate the partial result.
- Phase 3: Phase 1 and 2 alternates in an outer loop which loops over the K tiles. Once this outer loop finishes, each thread writes back to its assigned output elemenet.

We now need to determine the tile sizes of each dimension M, N, and K, which are often called BM, BN, and BK. To map the thread layout exactly to one output tile, BM will 
be 8 and BN will be 32 (suppose the thread layout is 32x8, and remember x-dim corresponds to the N dimension), and a common starting point for BK is 32 (warp size). 

Putting it all together, the CUDA pseudocode is

```c
__shared__ A_tile[BM][BK];
__shared__ B_tile[BK][BN];

int j = blockIdx.x * BN + threadIdx.x;
int i = blockIdx.y * BM + threadIdx.y;

float acc = 0f;
 
// Outer loop over K-tiles
for (int k = 0; k < K; k += BK) {
    // Threads load a BMxBK tile from A
    ...

    // Threads load a BKxBN tile from B
    ...


    // Sync is necessary because a thread block can have multiple warps
    __syncthreads(); 

    // Compute using data from shared memory
    for (int kk = 0; kk < BK; kk += 1) {
        acc += ...
    }

    // The second necessary sync because otherwise other warps may proceed to load the next tile
    __syncthreads(); 
}

// Write back
C[i*N + j] = acc;
``` 

For kernel luanch, we still use a 32x8 thread layout.

While this implementation alone won't achieve super high performance, it clearly shows the structure of shared memory tiling, and lays a foundation to explore increasing arith. intensity, as well as how that affects occupancy.

The idea of increaing arith. intensity is simply doing more compute (FMA) per memory load. We can achieve this because matmul inherently contains data reuse opportunities! Now we have two knobs to play with - the tile sizes (BM, BN, BK) and the thread layout e.g. 32x8, or 32x4 or 16x16 etc. Tile sizes determine compute per global memory load while thread layout (and the register reuse that comes with it) determines # number of compute per shared memory load. We introduce two additional symbols for per-thread micro-tile: TM and TN which represent the output tile that each thread computes. 

By tuning TM, TN and BM, BN we achieve different arithmetic intensity for shared memory and global memory respectively. In general, larger TM/TN achieves higher shared memory arithmetic intensity but also uses more registers per thread, while larger BM/BN achieves higher global memory arithmetic intensity but use more shared memory per thread block. Using more resources could lower the occupancy, so there's a tradeoff.


The simplest next step to explore data reuse is to keep the same tile sizes (BM=8, BN=32, BK=32), but change the thread layout to 32x4. This achives simple register reuse while shared memory usage stays the same. Alternatively, we can increase the tile sizes to BM=16, BN=32, BK=32 while keeping thread layout as 32x8 is. With these two examples we show how arithmetic intensity and occupancy are calculated.


## Case 1: BM=8, BN=32, BK=32, threads=32x4 => TM=2, TN=1

Each thread now computes two output element (two vertical elements), the total # of output elements computed per CTA is 8 * 32.

TileA: 8 * 32
TileB: 32 * 32
FMAs per tile: 8 * 32 * 32

Here BK is a common factor that can be eliminated.

So # of FMA per global memory load is 8 * 32 / (8 + 32) = 6.4, # of FMA per shared memory load is 2 / (2+1) = 0.66.

Basically the formula for global memory arith. intensity is BM * BN / (BM + BN) and for shared memory arith. intensity is TM * TN / (TM + TN).

## Case 2: BM=16, BN=32, BK=32, threads=32x8 => TM=2, TN=1

TM and TN are identical to case 1 so the # of FMA per shared memory load is the same. And the global memory arith. intensity is (16 * 32) / (16 + 32) = 10.66!

Wow a boost from previous 6.4! Why is that? Because 16x32 is more square-like than 8x32, and as we shall see later, a more square-like tile gives us higher arith. intensity.

However, now it comes to the tradeoff. Although case 2 achieve higher arithmetic intensity but it does use a 2x as large thread 
block with 256 threads and uses more shared memory (`16*32 + 32*32` vs case 1's `8*32 + 32*32`). But the increase in shared memory usage is less than 2x so the warp occupancy should be at least as good as case 1, or even better!  

Register and shared memory usage can be obtained just by compiler options `-Xptxas -v`. For RTX 4090, `nvcc -arch=sm_89 -O3 -Xptxas -v` would show the register and shared memory usage for your kernel.

|        | SMEM   | Register |
|--------|--------|----------|
| Case 1 | 2560B  | 48       |
| Case 2 | 3072B  | 40       |

Case 2 wins on both fronts — higher occupancy (few register and SMEM usage per thread) and higher arithmetic intensity.

# Stage 3: Square Tile Sizes via Tx-Remap
You've probably already noticed a tension: for coalesced memory access, we'd like the thread layout x-dim to be a multiple of 32 and accordingly the y-dim would typically be 4 or 8 because usually very large thread blocks (512 or 1024) could lower the occupancy; on the other hand, as we shall see soon, square tile sizes / micro-tile sizes achieve higher data reuse. The question is, how do we set BMxBN and TMxTN to be square-like while keeping the 32x4 or 32x8 physical thread layout? The trick here is a technique called `tx-remap`. 

The idea of `tx-remap` is to decouple data loading, computing and results write back. It assigns both a physical id (tx, ty) and a logical id to each thread (ltx, lty). 
While the physical layout must be 32x4 or 32x8, the logical layout can be organized however you like. For example, to create a 16x16 logical layout from a 32x8 physical layout,
you simply do:

1. Compute the flattened id for each thread: `int tid = ty * blockDim.x + tx`
2. Logical id is `int lty = tid / 16; int ltx = tid % 16`;

The physical id is used for cooperative loading, while the logical id is used for compute and write-back.

Once we have this remap technique in place, let's explore more BM, BN, TM, TN combinations. We can first fix TM and TN because they affect register usage.
To make register usage not too high, good candidates are 4x4, 4x8, 8x4. To push register reuse to an extreme, we can even try 8x8, though occupancy might be lower.

For a general occupancy consideration, we will use 128 or 256 threads per CTA with 32x4 or 32x8 physical layout. By multiplying TM/TN with the thread layout, we get BM, BN.
BK will stay 32 for now.

For example, the following configurations are all worth trying (thread layouts are logical layouts):

| TM | TN | Threads (logical)       | BM  | BN  | SMEM AI | Reg AI |
|----|----|----|-----|-----|---------|--------|
| 4  | 4  | 16x8  (128t, phys 32x4) | 32  | 64  | 21.33   | 2      |
| 4  | 4  | 16x16 (256t, phys 32x8) | 64  | 64  | 32      | 2      |
| 8  | 4  | 16x8  (128t, phys 32x4) | 64  | 64  | 32      | 2.67   |
| 8  | 8  | 16x8  (128t, phys 32x4) | 128 | 64  | 42.67   | 4      |
| 8  | 8  | 16x16 (256t, phys 32x8) | 128 | 128 | 64      | 4      |

Occupancy breakdown:

| TM | TN | Threads (logical)       | Regs/thread | SMEM/block | Active warps/SM | Occupancy |
|----|----|----|-------------|------------|-----------------|-----------|
| 4  | 4  | 16x8  (128t, phys 32x4) | 48          | 6144B      | 32              | 50%       |
| 4  | 4  | 16x16 (256t, phys 32x8) | 48          | 8192B      | 40              | 62.5%     |
| 8  | 4  | 16x8  (128t, phys 32x4) | 65          | 8192B      | 24              | 37.5%     |
| 8  | 8  | 16x8  (128t, phys 32x4) | 128         | 12288B     | 16              | 25%       |
| 8  | 8  | 16x16 (256t, phys 32x8) | 128         | 16384B     | 16              | 25%       |

Register and SMEM counts obtained by compiling with `nvcc -arch=sm_89 -O3 -Xptxas -v` on RTX 4090,
with `#pragma unroll 4` applied consistently to all tile-loading loops (loading loops with many
iterations must not be fully unrolled — doing so blows up register usage without benefit, since the
loop body is memory-latency-bound not instruction-count-bound).

Occupancy analysis (RTX 4090, SM 8.9):
- SM limits: 65536 registers, 48KB SMEM (default), 64 warps (2048 threads), 32 blocks max
- Register allocation granularity: 256 registers/warp → `ceil(regs × 32 / 256) × 256` per warp

| Config | Regs limiter | SMEM limiter | Binding |
|--------|-------------|--------------|---------|
| TM=4,TN=4, 128t | 10 blocks → 40w | 8 blocks → 32w | **SMEM** → 32w |
| TM=4,TN=4, 256t | 5 blocks → 40w | 6 blocks → 48w | **Regs** → 40w |
| TM=8,TN=4, 128t | 7 blocks → 28w | 6 blocks → 24w | **SMEM** → 24w |
| TM=8,TN=8, 128t | 4 blocks → 16w | 4 blocks → 16w | **Tie** → 16w  |
| TM=8,TN=8, 256t | 2 blocks → 16w | 3 blocks → 24w | **Regs** → 16w |

w = warps. Each row shows two limiters independently:                                                                 
  - "10 blocks → 40w" means: if only registers were the constraint, you could fit 10 blocks on the SM, giving 10 × 4 warps/block = 40 active warps                                            
  - "8 blocks → 32w" means: if only SMEM were the constraint, you could fit 8 blocks, giving 8 × 4 = 32 active warps                                                                          
  - Binding = whichever limit is tighter (smaller warp count wins)           

The register count is directly interpretable: TM×TN accumulators (float32) plus a small constant
for loop variables and indices. 16 accumulators → 48 regs, 32 → 65 regs, 64 → 128 regs.

As we increase BM/BN or TM/TN, data reuse keeps increasing. At TM=TN=8 with 16x16 logical layout,
reuse for both SMEM and register become very high. However, the 64 float32 accumulators per thread
(128 registers) halve the occupancy compared to TM=TN=4. The question is whether the higher
arithmetic intensity (21-64 global, 2-4 register) compensates for the lower occupancy.


The CUDA implementations for all 5 configs are in `mymatmul/gpu/_matmul_cuda_ext1.cu`.

## Empirical tuning: BK and tile-load unroll factor (RTX 4090, 4096³, bf16)

We swept BK ∈ {16, 32} and tile-load unroll ∈ {1, 2, 4, 8} for all 5 configs.
Peak GFLOPS at each (BK, unroll) combination:

```
BK=32:
┌─────────────────────┬──────────┬──────────┬──────────┬──────────┐
│       Kernel        │ unroll 1 │ unroll 2 │ unroll 4 │ unroll 8 │
├─────────────────────┼──────────┼──────────┼──────────┼──────────┤
│ tm4_tn4_bm32_bn64   │  29,262  │  22,148  │  24,233  │  30,098  │
│ tm4_tn4_bm64_bn64   │  31,087  │  26,369  │  28,008  │  32,080  │
│ tm8_tn4_bm64_bn64   │  33,176  │  27,853  │  29,117  │  33,867  │
│ tm8_tn8_bm128_bn64  │  18,566  │  29,870  │  35,233  │  35,872  │
│ tm8_tn8_bm128_bn128 │  39,970  │  34,298  │  38,901  │  39,804  │
└─────────────────────┴──────────┴──────────┴──────────┴──────────┘

BK=16:
┌─────────────────────┬──────────┬──────────┬──────────┬──────────┐
│       Kernel        │ unroll 1 │ unroll 2 │ unroll 4 │ unroll 8 │
├─────────────────────┼──────────┼──────────┼──────────┼──────────┤
│ tm4_tn4_bm32_bn64   │  19,183  │  22,777  │  25,872  │  30,144  │
│ tm4_tn4_bm64_bn64   │  24,210  │  26,392  │  32,888  │  32,293  │
│ tm8_tn4_bm64_bn64   │  24,273  │  27,632  │  29,983  │  35,494  │
│ tm8_tn8_bm128_bn64  │  21,353  │  29,288  │  34,675  │  35,408  │
│ tm8_tn8_bm128_bn128 │  26,271  │  33,854  │  38,083  │  35,890  │
└─────────────────────┴──────────┴──────────┴──────────┴──────────┘
```

**Takeaways:**
- **BK=32 is better across the board.** Even though BK=16 reduces SMEM usage and shortens
  the load loops, the larger K-tile gives more compute per __syncthreads() and better
  amortizes the synchronization overhead.
- **Unroll 8 is generally best.** More unrolling in the tile-load loops gives the compiler
  more ILP to hide global memory latency. The only exception is BK=32/unroll=1 which
  happens to be competitive for smaller configs (compiler chooses a good schedule on its
  own), and BK=32/tm8_tn8 configs where the A-load loop is 32 iters and unroll 8 vs
  unroll 1 are essentially tied — those kernels are compute-bound rather than load-bound.
- **Best config overall: tm8_tn8_bm128_bn128, BK=32, unroll=8 → ~40 TFLOPS**
  (27% of RTX 4090's 165 TFLOPS bf16 theoretical peak).



So increasing arithmetic intensity is more effective than increasing occupancy here because when 
the memory bandwidth is already saturated, increasing occupancy does not really help anymore.

Another question is, why does unrolling the cooperative tile loading loop help at all? And it 
seems more helpful to the high occupancy configurations than to the high a.i. ones
