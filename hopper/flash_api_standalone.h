/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 * Standalone Flash Attention API Header
 ******************************************************************************/

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Flash Attention 3 Standalone Forward Pass
 *
 * This is a standalone implementation without PyTorch dependencies.
 * All tensor pointers are passed as void*, and type information is passed via flags.
 *
 * @param q_ptr Query tensor device pointer
 * @param k_ptr Key tensor device pointer
 * @param v_ptr Value tensor device pointer
 * @param k_new_ptr Optional new keys for KV cache (nullptr if not used)
 * @param v_new_ptr Optional new values for KV cache (nullptr if not used)
 * @param q_v_ptr Optional Q projection for V (nullptr if not used)
 * @param out_ptr Output tensor device pointer (must be pre-allocated)
 * @param cu_seqlens_q_ptr Cumulative sequence lengths for Q (nullptr if not varlen)
 * @param cu_seqlens_k_ptr Cumulative sequence lengths for K (nullptr if not varlen)
 * @param cu_seqlens_k_new_ptr Cumulative sequence lengths for K_new (nullptr if not varlen)
 * @param seqused_q_ptr Sequence used for Q (nullptr if not used)
 * @param seqused_k_ptr Sequence used for K (nullptr if not used)
 * @param max_seqlen_q Maximum sequence length for Q (required if cu_seqlens_q is provided)
 * @param max_seqlen_k Maximum sequence length for K (required if cu_seqlens_k is provided)
 * @param page_table_ptr Paged KV cache page table (nullptr if not paged)
 * @param kv_batch_idx_ptr KV batch indices (nullptr if not used)
 * @param leftpad_k_ptr Left padding for K (nullptr if not used)
 * @param rotary_cos_ptr Rotary embedding cosine (nullptr if not used)
 * @param rotary_sin_ptr Rotary embedding sine (nullptr if not used)
 * @param seqlens_rotary_ptr Sequence lengths for rotary (nullptr if not used)
 * @param q_descale_ptr FP8 descale for Q (nullptr if not FP8)
 * @param k_descale_ptr FP8 descale for K (nullptr if not FP8)
 * @param v_descale_ptr FP8 descale for V (nullptr if not FP8)
 * @param scheduler_metadata_ptr Scheduler metadata (nullptr to allocate internally)
 * @param is_bf16 True if using BF16, false if FP16
 * @param is_e4m3 True if using FP8 E4M3
 * @param batch_size_q Batch size for Q
 * @param seqlen_q Sequence length for Q (or total_q if varlen)
 * @param num_heads Number of query heads
 * @param head_size Head dimension
 * @param batch_size_k Batch size for K
 * @param seqlen_k Sequence length for K (or total_k if varlen, or num_pages if paged)
 * @param num_heads_k Number of key/value heads
 * @param head_size_v Head dimension for V
 * @param page_size Page size (only used if paged)
 * @param max_num_pages_per_seq Maximum number of pages per sequence (only used if paged)
 * @param seqlen_k_new Length of K_new (only used if k_new_ptr != nullptr)
 * @param rotary_dim Rotary embedding dimension (only used if rotary_cos_ptr != nullptr)
 * @param q_batch_stride Q batch stride
 * @param q_row_stride Q row stride
 * @param q_head_stride Q head stride
 * @param k_batch_stride K batch stride
 * @param k_row_stride K row stride
 * @param k_head_stride K head stride
 * @param v_batch_stride V batch stride
 * @param v_row_stride V row stride
 * @param v_head_stride V head stride
 * @param o_batch_stride Output batch stride
 * @param o_row_stride Output row stride
 * @param o_head_stride Output head stride
 * @param k_new_batch_stride K_new batch stride
 * @param k_new_row_stride K_new row stride
 * @param k_new_head_stride K_new head stride
 * @param v_new_batch_stride V_new batch stride
 * @param v_new_row_stride V_new row stride
 * @param v_new_head_stride V_new head stride
 * @param q_v_batch_stride Q_v batch stride
 * @param q_v_row_stride Q_v row stride
 * @param q_v_head_stride Q_v head stride
 * @param page_table_batch_stride Page table batch stride
 * @param q_descale_batch_stride Q descale batch stride
 * @param q_descale_head_stride Q descale head stride
 * @param k_descale_batch_stride K descale batch stride
 * @param k_descale_head_stride K descale head stride
 * @param v_descale_batch_stride V descale batch stride
 * @param v_descale_head_stride V descale head stride
 * @param softmax_scale_val Softmax scale (< 0 to use default 1/sqrt(head_size))
 * @param is_causal Whether to apply causal masking
 * @param window_size_left Window size on the left (-1 for infinite)
 * @param window_size_right Window size on the right (-1 for infinite)
 * @param attention_chunk Attention chunk size
 * @param softcap Softcap value
 * @param is_rotary_interleaved Whether rotary embedding is interleaved
 * @param num_splits Number of splits (<= 0 to auto-determine)
 * @param pack_gqa_val Pack GQA value (< 0 to auto-determine)
 * @param sm_margin SM margin
 * @param softmax_lse_ptr Softmax LSE output (must be pre-allocated)
 * @param out_accum_ptr Output accumulator (nullptr if num_splits == 1)
 * @param softmax_lse_accum_ptr Softmax LSE accumulator (nullptr if num_splits == 1)
 * @param stream CUDA stream
 */
void mha_fwd_standalone(
        // Q, K, V device pointers
        void* q_ptr,
        void* k_ptr,
        void* v_ptr,
        // Optional k_new, v_new (for KV cache appending)
        void* k_new_ptr,
        void* v_new_ptr,
        // Optional q_v
        void* q_v_ptr,
        // Output (can be pre-allocated or caller allocates)
        void* out_ptr,
        // Variable length support
        void* cu_seqlens_q_ptr,
        void* cu_seqlens_k_ptr,
        void* cu_seqlens_k_new_ptr,
        void* seqused_q_ptr,
        void* seqused_k_ptr,
        int64_t max_seqlen_q,
        int64_t max_seqlen_k,
        // Paged KV support
        void* page_table_ptr,
        void* kv_batch_idx_ptr,
        // Leftpad support
        void* leftpad_k_ptr,
        // Rotary embeddings
        void* rotary_cos_ptr,
        void* rotary_sin_ptr,
        void* seqlens_rotary_ptr,
        // FP8 descale
        void* q_descale_ptr,
        void* k_descale_ptr,
        void* v_descale_ptr,
        // Scheduler metadata (optional pre-allocated)
        void* scheduler_metadata_ptr,
        // Type flags
        bool is_bf16,
        bool is_e4m3,
        // Dimensions for Q
        int batch_size_q,
        int seqlen_q,
        int num_heads,
        int head_size,
        // Dimensions for K/V
        int batch_size_k,
        int seqlen_k,
        int num_heads_k,
        int head_size_v,
        // Paged KV dimensions
        int page_size,
        int max_num_pages_per_seq,
        // k_new/v_new dimensions
        int seqlen_k_new,
        // Rotary dimensions
        int rotary_dim,
        // Q strides
        int64_t q_batch_stride,
        int64_t q_row_stride,
        int64_t q_head_stride,
        // K strides
        int64_t k_batch_stride,
        int64_t k_row_stride,
        int64_t k_head_stride,
        // V strides
        int64_t v_batch_stride,
        int64_t v_row_stride,
        int64_t v_head_stride,
        // Output strides
        int64_t o_batch_stride,
        int64_t o_row_stride,
        int64_t o_head_stride,
        // k_new/v_new strides
        int64_t k_new_batch_stride,
        int64_t k_new_row_stride,
        int64_t k_new_head_stride,
        int64_t v_new_batch_stride,
        int64_t v_new_row_stride,
        int64_t v_new_head_stride,
        // q_v strides
        int64_t q_v_batch_stride,
        int64_t q_v_row_stride,
        int64_t q_v_head_stride,
        // Page table strides
        int64_t page_table_batch_stride,
        // FP8 descale strides
        int64_t q_descale_batch_stride,
        int64_t q_descale_head_stride,
        int64_t k_descale_batch_stride,
        int64_t k_descale_head_stride,
        int64_t v_descale_batch_stride,
        int64_t v_descale_head_stride,
        // Attention parameters
        double softmax_scale_val,
        bool is_causal,
        int64_t window_size_left,
        int64_t window_size_right,
        int64_t attention_chunk,
        double softcap,
        bool is_rotary_interleaved,
        int64_t num_splits,
        int pack_gqa_val,
        int64_t sm_margin,
        // Pre-allocated output buffers (MUST provide)
        void* softmax_lse_ptr,
        void* out_accum_ptr,
        void* softmax_lse_accum_ptr,
        // CUDA stream
        cudaStream_t stream
);

#ifdef __cplusplus
}
#endif
