import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=1),
    ],
    key=['n_cols'],
)
@triton.jit
def softmax_kernel_tiled(
    input_ptr,
    output_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * n_cols
    output_row_ptr = output_ptr + row_idx * n_cols
    
    # First pass: find max using tiled reduction
    row_max = float('-inf')
    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        block_data = tl.load(row_start_ptr + col_offsets, mask=mask, other=float('-inf'))
        block_max = tl.max(block_data, axis=0)
        row_max = tl.maximum(row_max, block_max)
    
    # Second pass: compute sum of exp(x - max) using tiled reduction
    sum_exp = 0.0
    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        block_data = tl.load(row_start_ptr + col_offsets, mask=mask, other=0.0)
        block_exp = tl.exp(block_data - row_max)
        sum_exp += tl.sum(block_exp, axis=0)
    
    # Third pass: compute and store softmax values
    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        block_data = tl.load(row_start_ptr + col_offsets, mask=mask, other=0.0)
        softmax_output = tl.exp(block_data - row_max) / sum_exp
        tl.store(output_row_ptr + col_offsets, softmax_output, mask=mask)


def run(x):
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)
    
    grid = (n_rows,)
    softmax_kernel_tiled[grid](
        x,
        output,
        n_cols,
    )
    
    return output
