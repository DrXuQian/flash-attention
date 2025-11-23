# Flash Attention 3 Standalone - Usage Guide

## Build Configuration

This standalone build supports:
- **FP8 E4M3**: head_dim ≤ 96 (compiled kernels use hdim=96)
- **FP16**: head_dim ≤ 128 (compiled kernels use hdim=128)
- **BF16**: NOT supported

Features included:
- ✅ Variable-length sequences (varlen)
- ✅ Group Query Attention (GQA)
- ✅ Causal attention
- ✅ Split-KV
- ✅ PackGQA
- ✅ Softcap
- ✅ Paged KV cache

## Building

```bash
cd /path/to/flash-attention/hopper
bash build_standalone.sh
```

The compiled executable will be at: `build_standalone/test_varlen_inference`

## Supported Test Cases

All four production cases are fully supported:

### Case 1: FP8 Large Batch Prefill
```bash
./build_standalone/test_varlen_inference \
    --fp8 \
    --batch_size 3 \
    --max_seqlen_q 1680 \
    --max_seqlen_k 1680 \
    --num_heads 16 \
    --num_heads_k 16 \
    --head_size 80
```

**Details:**
- Data type: FP8 E4M3
- Batch size: 3
- Sequence length: 1680 (both Q and K)
- Heads: 16 query heads, 16 KV heads (no GQA)
- Head dimension: 80 (padded to 96 internally)
- Causal: No

### Case 2: FP8 Small Batch High Throughput
```bash
./build_standalone/test_varlen_inference \
    --fp8 \
    --batch_size 84 \
    --max_seqlen_q 64 \
    --max_seqlen_k 64 \
    --num_heads 16 \
    --num_heads_k 16 \
    --head_size 80
```

**Details:**
- Data type: FP8 E4M3
- Batch size: 84 (high batch parallelism)
- Sequence length: 64 (short sequences)
- Heads: 16 query heads, 16 KV heads
- Head dimension: 80 (padded to 96 internally)
- Causal: No

### Case 3: FP16 GQA Prefill with Causal Masking
```bash
./build_standalone/test_varlen_inference \
    --fp16 \
    --batch_size 1 \
    --max_seqlen_q 1285 \
    --max_seqlen_k 1285 \
    --num_heads 16 \
    --num_heads_k 2 \
    --head_size 80 \
    --causal
```

**Details:**
- Data type: FP16
- Batch size: 1
- Sequence length: 1285 (both Q and K)
- Heads: 16 query heads, 2 KV heads (GQA with ratio 8:1)
- Head dimension: 80 (padded to 128 internally)
- Causal: Yes (autoregressive generation)

### Case 4: FP16 GQA Decode with Causal Masking
```bash
./build_standalone/test_varlen_inference \
    --fp16 \
    --batch_size 1 \
    --max_seqlen_q 1 \
    --max_seqlen_k 2048 \
    --num_heads 16 \
    --num_heads_k 2 \
    --head_size 80 \
    --causal
```

**Details:**
- Data type: FP16
- Batch size: 1
- Sequence length: Q=1 (decode step), K=2048 (context)
- Heads: 16 query heads, 2 KV heads (GQA with ratio 8:1)
- Head dimension: 80 (padded to 128 internally)
- Causal: Yes (typical decode scenario)

## Command Line Options

```
Usage: test_varlen_inference [options]

Options:
  --batch_size N       Batch size (default: 2)
  --max_seqlen_q N     Max sequence length for Q (default: 512)
  --max_seqlen_k N     Max sequence length for K (default: 512)
  --num_heads N        Number of query heads (default: 8)
  --num_heads_k N      Number of key/value heads (default: 8)
  --head_size N        Head dimension (default: 128 for FP16, use 96 for FP8)
  --causal             Use causal attention (default: false)
  --fp16               Use FP16 (default)
  --fp8                Use FP8 E4M3 (requires head_size <= 96)
  --help               Show this help message
```

## Technical Notes

### Head Dimension Padding

The kernels are compiled with specific head dimensions (96 for FP8, 128 for FP16), but they can handle smaller dimensions through padding:

- **FP8**: head_dim=80 → padded to 96 internally
- **FP16**: head_dim=80 → padded to 128 internally

The padding is transparent to the user and only uses the actual head_dim values in computation.

### Group Query Attention (GQA)

GQA is automatically enabled when `num_heads != num_heads_k`. The implementation uses PackGQA kernels for optimal performance:

- `num_heads / num_heads_k` must be a power of 2
- Common ratios: 2:1, 4:1, 8:1, 16:1

### Variable-Length Sequences

The test program generates random sequence lengths within the specified `max_seqlen_q` and `max_seqlen_k` bounds. The varlen attention uses cumulative sequence length (cu_seqlens) format internally.

### Memory Requirements

- **Scheduler Metadata**: Automatically allocated (1024 int32 values = 4KB)
- **Softmax LSE**: `num_heads * total_q * sizeof(float)`
- **Q/K/V/O buffers**: Sized according to total tokens across batch

## Performance Expectations

The test program reports:
- Average execution time (ms)
- Throughput (TFLOP/s)

Typical performance on H100:
- **FP8**: ~300-400 TFLOP/s
- **FP16**: ~150-250 TFLOP/s

Performance varies based on:
- Sequence length (longer sequences are more efficient)
- Batch size (larger batches amortize overhead)
- GQA ratio (affects memory bandwidth)
- Causal vs non-causal (causal has less computation)

## Troubleshooting

### Illegal Memory Access

If you see "illegal memory access" errors:
- Ensure scheduler_metadata is allocated (fixed in latest version)
- Check GPU memory availability
- Verify CUDA architecture matches (SM90 required)

### Assertion Failures

- **"head_dim <= 96 for FP8"**: Use `--head_size 96` or smaller
- **"head_dim <= 128 for FP16"**: Use `--head_size 128` or smaller
- **"BF16 not supported"**: Use `--fp16` or `--fp8`, not `--bf16`

### Compile Errors

- Ensure CUDA 12.x is installed
- Check that cutlass submodule is initialized
- Use `make -j4` instead of higher parallelism if running out of memory

## API Usage (Advanced)

To use the standalone API in your own code:

```cpp
#include "flash_api_standalone.h"

// Prepare input data (Q, K, V as void* pointers)
// Allocate scheduler_metadata on GPU
void* d_scheduler_metadata;
cudaMalloc(&d_scheduler_metadata, 1024 * sizeof(int));
cudaMemset(d_scheduler_metadata, 0, 1024 * sizeof(int));

// Call the API
mha_fwd_standalone(
    d_q, d_k, d_v,
    nullptr, nullptr, nullptr,  // k_new, v_new, q_v
    d_out,
    d_cu_seqlens_q, d_cu_seqlens_k, nullptr,
    nullptr, nullptr,  // seqused_q, seqused_k
    max_seqlen_q, max_seqlen_k,
    nullptr, nullptr, nullptr,  // page_table, kv_batch_idx, leftpad_k
    nullptr, nullptr, nullptr,  // rotary_cos, rotary_sin, seqlens_rotary
    nullptr, nullptr, nullptr,  // q_descale, k_descale, v_descale
    d_scheduler_metadata,
    is_bf16, is_e4m3,
    batch_size, total_q, num_heads, head_size,
    batch_size, total_k, num_heads_k, head_size_v,
    0, 0, 0, 0,  // paged KV, rotary dim
    q_batch_stride, q_row_stride, q_head_stride,
    k_batch_stride, k_row_stride, k_head_stride,
    v_batch_stride, v_row_stride, v_head_stride,
    o_batch_stride, o_row_stride, o_head_stride,
    0, 0, 0, 0, 0, 0,  // k_new, v_new strides
    0, 0, 0,  // q_v strides
    0,  // page_table stride
    0, 0, 0, 0, 0, 0,  // descale strides
    -1.0,  // softmax_scale (auto)
    is_causal,
    -1, -1,  // window_size (no local attention)
    0,  // attention_chunk
    0.0,  // softcap
    false,  // is_rotary_interleaved
    -1, -1, 0,  // num_splits (auto), pack_gqa (auto), sm_margin
    d_softmax_lse,
    nullptr, nullptr,  // out_accum, softmax_lse_accum
    stream
);

// Cleanup
cudaFree(d_scheduler_metadata);
```

## Build Configuration Details

Compiled kernel variants:
- **FP16 hdim=128**: ~10 variants (Split/PackGQA/Softcap/Paged combinations)
- **FP8 hdim=96**: ~10 variants (Split/PackGQA/Softcap/Paged combinations)

Total compilation time: 1-2 minutes
Memory requirement: ~8GB RAM

Compilation flags:
- `-DFLASHATTENTION_DISABLE_BACKWARD`: No backward pass
- `-DFLASHATTENTION_DISABLE_SM8x`: SM90 only (no SM80/SM86)
- `-O3`: Full optimization
- `--expt-relaxed-constexpr`: CUTLASS compatibility
