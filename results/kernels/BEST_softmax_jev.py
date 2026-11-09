import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16, num_stages=1),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16, num_stages=1),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=1),
    ],
    key=['n_cols'],
)
@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row
    row_idx = tl.program_id(0)
    
    # Compute the starting pointer for this row
    row_start_ptr = input_ptr + row_idx * n_cols
    output_row_ptr = output_ptr + row_idx * n_cols
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Pass 1: Find max
    row_max = float('-inf')
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offsets = block_start + col_offsets
        mask = offsets < n_cols
        row_block = tl.load(row_start_ptr + offsets, mask=mask, other=float('-inf'))
        block_max = tl.max(row_block)
        row_max = tl.maximum(row_max, block_max)
    
    # Pass 2: Compute exp and sum
    exp_sum = 0.0
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offsets = block_start + col_offsets
        mask = offsets < n_cols
        row_block = tl.load(row_start_ptr + offsets, mask=mask, other=0.0)
        exp_block = tl.exp(row_block - row_max)
        tl.store(output_row_ptr + offsets, exp_block, mask=mask)
        exp_sum += tl.sum(tl.where(mask, exp_block, 0.0))
    
    # Pass 3: Normalize
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offsets = block_start + col_offsets
        mask = offsets < n_cols
        exp_block = tl.load(output_row_ptr + offsets, mask=mask, other=0.0)
        softmax_block = exp_block / exp_sum
        tl.store(output_row_ptr + offsets, softmax_block, mask=mask)


def run(x):
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)
    
    grid = (n_rows,)
    
    softmax_kernel[grid](
        x,
        output,
        n_cols,
    )
    
    return output
