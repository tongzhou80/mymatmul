import torch


def matmul_cuda_naive_ijk(A_gpu, B_gpu):
    """Naive CUDA matmul: 2D grid of threads (i->blockIdx.x, j->blockIdx.y).
    Each thread (i,j) computes full k reduction.

    Assumes A_gpu and B_gpu are already on GPU.
    Returns result as GPU tensor.
    """
    from . import _matmul_cuda_ext
    return _matmul_cuda_ext.matmul_cuda_naive_ijk(A_gpu, B_gpu)


def matmul_cuda_naive_ijk_jx(A_gpu, B_gpu):
    """Naive CUDA matmul: 2D grid of threads (j->blockIdx.x, i->blockIdx.y).
    Maps j (column, fastest-changing in row-major) to x (fastest CUDA dimension).
    Each thread (i,j) computes full k reduction.

    Assumes A_gpu and B_gpu are already on GPU.
    Returns result as GPU tensor.
    """
    from . import _matmul_cuda_ext
    return _matmul_cuda_ext.matmul_cuda_naive_ijk_jx(A_gpu, B_gpu)


def matmul_cuda_tiled_32x32(A_gpu, B_gpu):
    """Tiled CUDA matmul: 32x32 tiles with shared memory.
    Processes K dimension in 32-element tiles, loading into shared memory.
    Each thread (i,j) maintains its own running sum across all K-tiles.
    Thread block: 32x32 = 1024 threads.

    Assumes A_gpu and B_gpu are already on GPU.
    Returns result as GPU tensor.
    """
    from . import _matmul_cuda_ext
    return _matmul_cuda_ext.matmul_cuda_tiled_32x32(A_gpu, B_gpu)


def matmul_cuda_tiled_32x32_threads_16x16(A_gpu, B_gpu):
    """Optimized tiled CUDA matmul: 32x32 tiles with 16x16 threads.

    Thread mapping:
    - Block computes C tile [block_row : block_row+32, block_col : block_col+32]
    - Each thread computes a 2x2 micro-tile with 4 accumulator registers

    Memory access pattern:
    - Each thread loads 4 elements into A_shared and 4 into B_shared
    - Shared memory: A_shared[32][32] + B_shared[32][32]

    Benefits:
    - Reduced synchronization overhead (256 vs 1024 threads)
    - Same data reuse from shared memory (32x)
    - Better register usage per thread
    - More instruction-level parallelism (4 independent accumulators)
    - 2.4x faster than 32x32 threads version

    Assumes A_gpu and B_gpu are already on GPU and contiguous.
    Returns result as GPU tensor.
    """
    from . import _matmul_cuda_ext
    return _matmul_cuda_ext.matmul_cuda_tiled_32x32_16x16_threads(A_gpu, B_gpu)


def matmul_cuda_tiled_32x32_threads_32x8(A_gpu, B_gpu):
    """Warp-aligned tiled CUDA matmul: 32x32 tiles with 32x8 threads.

    Thread mapping:
    - Blocksize: 32x8 = 256 threads (8 warps)
    - Block computes C tile [block_row : block_row+32, block_col : block_col+32]
    - Each thread computes a 4x1 micro-tile (4 rows, 1 column)

    Warp alignment:
    - One warp (32 threads) operates on one row stripe of the output tile
    - Each thread (tx, ty) computes 4 consecutive rows with tx as the column

    Memory access pattern:
    - Linearized loading: each thread loads 4 elements from A and B
    - Shared memory: A_shared[32][32] + B_shared[32][32]

    Assumes A_gpu and B_gpu are already on GPU and contiguous.
    Returns result as GPU tensor.
    """
    from . import _matmul_cuda_ext
    return _matmul_cuda_ext.matmul_cuda_tiled_32x32_32x8_threads(A_gpu, B_gpu)


def matmul_cuda_tiled_32x32_threads_32x4(A_gpu, B_gpu):
    """Ultra-warp-aligned tiled CUDA matmul: 32x32 tiles with 32x4 threads.

    Thread mapping:
    - Blocksize: 32x4 = 128 threads (4 warps)
    - Block computes C tile [block_row : block_row+32, block_col : block_col+32]
    - Each thread computes an 8x1 micro-tile (8 rows, 1 column)

    Warp alignment:
    - Each warp operates on 8 consecutive rows of the output tile
    - Extreme thread efficiency: each of 32 threads in warp handles one output column

    Memory access pattern:
    - Linearized loading: each thread loads 8 elements from A and B (1024 / 128 = 8)
    - Shared memory: A_shared[32][32] + B_shared[32][32]
    - Loop unrolled loads with #pragma unroll

    Assumes A_gpu and B_gpu are already on GPU and contiguous.
    Returns result as GPU tensor.
    """
    from . import _matmul_cuda_ext
    return _matmul_cuda_ext.matmul_cuda_tiled_32x32_32x4_threads(A_gpu, B_gpu)  # Calls the refactored kernel


def matmul_cuda_tiled_32x64_tm4_tn4(A_gpu, B_gpu):
    """Symmetric tiled CUDA matmul: 32x64 output tiles with 32x4 threads remapped to 8x16 logical grid.

    Each thread computes a 4x4 micro-tile (TM=TN=4) using tid-based remapping
    to decouple physical layout from logical output assignment.

    Physical layout:    32x4  = 128 threads
    Logical grid:        8x16 = 128 threads (BM/TM x BN/TN = 32/4 x 64/4)
    Thread tile:         TM=4, TN=4 = 16 outputs per thread

    Assumes A_gpu and B_gpu are already on GPU and contiguous.
    Returns result as GPU tensor.
    """
    from . import _matmul_cuda_ext
    return _matmul_cuda_ext.matmul_cuda_tiled_32x64_tm4_tn4(A_gpu, B_gpu)


def matmul_cuda_tiled_32x64_threads_32x4(A_gpu, B_gpu):
    """Larger tile CUDA matmul: 32x64 output tiles with 32x4 threads.

    Thread mapping:
    - Blocksize: 32x4 = 128 threads (same as 32x32 version)
    - Block computes C tile [block_row : block_row+32, block_col : block_col+64]
    - Each thread computes an 8x2 micro-tile (8 rows, 2 columns)

    Output mapping:
    - Thread (tx, ty) computes columns [tx, tx+32] for 8 consecutive rows
    - Rows stride by 4 per thread: row offsets [ty, ty+4, ty+8, ..., ty+28]
    - This gives 2D spatial locality with good cache utilization

    Memory architecture:
    - Shared memory: A_shared[32][32] + B_shared[32][64] = 3072 elements
    - Each thread loads 8 elements from A, 16 from B per K-tile
    - Larger B_shared footprint for better column reuse

    Benefits over 32x32 tiles:
    - More work per block in N dimension → better amortization
    - Each thread computes 16 elements in 8x2 arrangement
    - Better cache locality from 2D output pattern
    - More computation before synchronization

    Trade-offs:
    - Higher shared memory usage (3072 vs 2048 bytes)
    - Fewer blocks per SM if memory-limited, more work per block
    - Increased B_shared bandwidth but coalescing still good

    Assumes A_gpu and B_gpu are already on GPU and contiguous.
    Returns result as GPU tensor.
    """
    from . import _matmul_cuda_ext
    return _matmul_cuda_ext.matmul_cuda_tiled_32x64_32x4_threads(A_gpu, B_gpu)
