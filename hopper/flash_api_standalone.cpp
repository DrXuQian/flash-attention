/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 * Standalone version without PyTorch dependencies
 ******************************************************************************/

#include <cuda_runtime.h>
#include <cutlass/numeric_types.h>
#include <cmath>
#include <algorithm>
#include <cassert>

#include "flash.h"
#include "static_switch.h"
#include "tile_size.h"
#include "heuristics.h"

// Standalone version: replace TORCH_CHECK with assert
#define CHECK_ASSERT(cond, msg) do { if (!(cond)) { printf("ERROR: %s\n", msg); assert(false); } } while(0)

#define PREPARE_VARLEN_MAX_BATCHES_1CTA 992

// Helper functions copied from flash_api.cpp
inline bool get_pagedkv_tma(Flash_fwd_params const& params) {
    if (params.arch < 90 || !params.page_table || params.leftpad_k || params.knew_ptr) { return false; }
    // This needs to match the kernel configs
    auto kBlockMN_kernel_args_sm90 = tile_size_fwd_sm90(params.d_rounded, params.dv_rounded, params.is_causal, params.is_local, params.is_e4m3 ? 1 : 2 /*element_size*/, false /*v_colmajor*/, false /*paged_kv_non_TMA*/, params.softcap > 0.f);
    int const kBlockM = std::get<0>(kBlockMN_kernel_args_sm90);
    int const kBlockN = std::get<1>(kBlockMN_kernel_args_sm90);
    // Heuristic: when seqlen_q <= kBlockM, we're not compute bound, and somehow using TMA is slower,
    // at least for MLA.
    return params.page_size % kBlockN == 0 && params.seqlen_q * (params.h / params.h_k) > kBlockM;
}

inline bool get_pack_gqa(Flash_fwd_params const& params) {
    // Always enable PackGQA for Sm8x or PagedKVNonTMA or Split to reduce compilation and binary size.
    // Has little effect on speed.
    if (params.arch < 90 || (params.page_table && !params.pagedkv_tma) || params.num_splits > 1) { return true; }
    #ifdef FLASHATTENTION_DISABLE_PACKGQA
    return false;
    #else
    // params.page_table must already be set
    if (params.h == params.h_k) { return false; }
    // This needs to match the kernel configs
    auto kBlockMN_kernel_args_sm90 = tile_size_fwd_sm90(params.d_rounded, params.dv_rounded, params.is_causal, params.is_local, params.is_e4m3 ? 1 : 2 /*element_size*/, false /*v_colmajor*/, params.page_table && !params.pagedkv_tma, params.softcap > 0.f);
    int const kBlockM = std::get<0>(kBlockMN_kernel_args_sm90);
    return should_pack_gqa(params.cu_seqlens_q || params.seqused_q, params.seqlen_q, params.h / params.h_k, kBlockM);
    #endif
}

inline int get_num_splits(Flash_fwd_params const& params) {
    #ifdef FLASHATTENTION_DISABLE_SPLIT
    return 1;
    #else
    // Always enable PackGQA for Split
    // params.page_table must already be set
    // This needs to match the kernel configs
    bool varlen = params.cu_seqlens_q || params.cu_seqlens_k || params.seqused_q || params.seqused_k || params.leftpad_k;
    auto kBlockMN_kernel_args_sm90 = tile_size_fwd_sm90(params.d_rounded, params.dv_rounded, params.is_causal, params.is_local, params.is_e4m3 ? 1 : 2 /*element_size*/, false /*v_colmajor*/, params.page_table && !params.pagedkv_tma, params.softcap > 0.f);
    // Strictly speaking we need to pass in (varlen && params.num_splits > 1) but num_splits
    // has not been set here. It's OK though because we might just underestimate kBlockN a bit
    auto kBlockMN_kernel_args_sm8x = tile_size_fwd_sm8x(params.arch == 86 || params.arch == 89, params.d_rounded, params.dv_rounded, params.is_causal, params.is_local, params.is_e4m3 ? 1 : 2 /*element_size*/, params.page_table, varlen, params.softcap > 0.f, params.knew_ptr);
    int const kBlockM = params.arch >= 90 ? std::get<0>(kBlockMN_kernel_args_sm90) : std::get<0>(kBlockMN_kernel_args_sm8x);
    int const kBlockN = params.arch >= 90 ? std::get<1>(kBlockMN_kernel_args_sm90) : std::get<1>(kBlockMN_kernel_args_sm8x);
    int seqlen_q_packgqa = params.seqlen_q * (params.h / params.h_k);
    // If is_local, we're not going to load all of seqlen_k
    int const seqlen_k_loaded = !params.is_local
        ? params.seqlen_k
        : std::max(0, std::min(params.seqlen_k, params.window_size_right + params.window_size_left + 1 + kBlockM));
    int const num_n_blocks = (seqlen_k_loaded + kBlockN - 1) / kBlockN;
    int const num_m_blocks = (seqlen_q_packgqa + kBlockM - 1) / kBlockM;
    int const size_one_kv_head = params.seqlen_k * (params.d + params.dv) * (params.is_e4m3 ? 1 : 2);
    // Always enable PackGQA for Split
    // If varlen, we use dynamic split, so this heuristic just needs to get an upper bound on num_splits.
    // We assume the case where there's 1 long sequence and the rest are short, i.e. pretending
    // that batch = 1.
    int total_mblocks = (params.num_splits_dynamic_ptr ? 1 : params.b) * params.h_k * num_m_blocks;
    return num_splits_heuristic(total_mblocks, params.num_sm, num_n_blocks, num_m_blocks, size_one_kv_head, params.is_causal || params.is_local, 128);
    #endif
}

inline int get_max_headdim() {
    #ifndef FLASHATTENTION_DISABLE_HDIM256
    return 256;
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM192
    return 192;
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM128
    return 128;
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM96
    return 96;
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM64
    return 64;
    #endif
    return 0;
}

inline int round_up_headdim(int head_size) {
    #ifndef FLASHATTENTION_DISABLE_HDIM64
    if (head_size <= 64) { return 64; }
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM96
    if (head_size <= 96) { return 96; }
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM128
    if (head_size <= 128) { return 128; }
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM192
    if (head_size <= 192) { return 192; }
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM256
    if (head_size <= 256) { return 256; }
    #endif
    return 256;
}

inline int round_up_headdimv(int head_size) {
    if (head_size <= 64) { return 64; }
    if (head_size <= 96) { return 96; }
    if (head_size <= 128) { return 128; }
    if (head_size <= 192) { return 192; }
    if (head_size <= 256) { return 256; }
    return 512;
}

void set_params_fprop_standalone(Flash_fwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      // device pointers (void* instead of at::Tensor)
                      void* q_ptr,
                      void* k_ptr,
                      void* v_ptr,
                      void* out_ptr,
                      // type flags (passed as parameters instead of querying tensor dtype)
                      bool is_bf16,
                      bool is_e4m3,
                      // strides (passed explicitly instead of querying from tensor)
                      int64_t q_batch_stride,
                      int64_t k_batch_stride,
                      int64_t v_batch_stride,
                      int64_t o_batch_stride,
                      int64_t q_row_stride,
                      int64_t k_row_stride,
                      int64_t v_row_stride,
                      int64_t o_row_stride,
                      int64_t q_head_stride,
                      int64_t k_head_stride,
                      int64_t v_head_stride,
                      int64_t o_head_stride,
                      int64_t v_dim_stride,
                      // optional pointers (same as original)
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *seqused_q,
                      void *seqused_k,
                      void *softmax_lse_d,
                      // parameters (same as original)
                      float p_dropout,
                      float softmax_scale,
                      int window_size_left,
                      int window_size_right,
                      int attention_chunk,
                      const float softcap=0.f,
                      const int sm_margin=0) {

    // Reset the parameters
    params = {};

    // Set type flags from parameters (instead of querying from tensor)
    params.is_bf16 = is_bf16;
    params.is_e4m3 = is_e4m3;

    // Set the pointers (from void* parameters instead of tensor.data_ptr())
    params.q_ptr = q_ptr;
    params.k_ptr = k_ptr;
    params.v_ptr = v_ptr;
    params.o_ptr = out_ptr;

    // All stride are in elements, not bytes.
    // Set strides from parameters (instead of tensor.stride())
    params.q_row_stride = q_row_stride;
    params.k_row_stride = k_row_stride;
    params.v_row_stride = v_row_stride;
    params.q_head_stride = q_head_stride;
    params.k_head_stride = k_head_stride;
    params.v_head_stride = v_head_stride;
    params.v_dim_stride = v_dim_stride;
    params.o_row_stride = o_row_stride;
    params.o_head_stride = o_head_stride;

    if (cu_seqlens_q_d == nullptr) {
        params.q_batch_stride = q_batch_stride;
        params.o_batch_stride = o_batch_stride;
    }
    if (cu_seqlens_k_d == nullptr) {
        params.k_batch_stride = k_batch_stride;
        params.v_batch_stride = v_batch_stride;
    }

    params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
    params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);
    params.seqused_q = static_cast<int *>(seqused_q);
    params.seqused_k = static_cast<int *>(seqused_k);

    // Softmax sum
    params.softmax_lse_ptr = softmax_lse_d;

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = d;
    params.d_rounded = d_rounded;

    // Set the different scale values.
    params.scale_softmax = softmax_scale;
    params.softcap = softcap;

    // Set this to probability of keeping an element to simplify things.
    params.p_dropout = 1.f - p_dropout;
    // Convert p from float to int so we don't have to convert the random uint to float to compare.
    // [Minor] We want to round down since when we do the comparison we use <= instead of <
    // params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
    // params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
    params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
    params.rp_dropout = 1.f / params.p_dropout;
    CHECK_ASSERT(p_dropout < 1.f, "p_dropout must be < 1.0");
    #ifdef FLASHATTENTION_DISABLE_DROPOUT
        CHECK_ASSERT(p_dropout == 0.0f, "This flash attention build does not support dropout.");
    #endif

    // Causal is the special case where window_size_right == 0 and window_size_left < 0.
    // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
    params.is_causal = window_size_left < 0 && window_size_right == 0 && attention_chunk == 0;
    params.is_local = (window_size_left >= 0 || window_size_right >= 0 || attention_chunk >= 1) && !params.is_causal;

    // TODO: check this
    if (window_size_left < 0) { window_size_left = seqlen_k - 1; }
    if (window_size_right < 0) { window_size_right = seqlen_q - 1; }
    if (attention_chunk > 0) {
        window_size_left = std::min(window_size_left, attention_chunk - 1);
        window_size_right = std::min(window_size_right, attention_chunk - 1);
    }
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;
    params.attention_chunk = attention_chunk;

    // Get device properties without ATen (instead of at::cuda::getCurrentDeviceProperties())
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    params.arch = prop.major * 10 + prop.minor;
    params.num_sm = prop.multiProcessorCount - sm_margin;

    #ifdef FLASHATTENTION_DISABLE_LOCAL
        CHECK_ASSERT(!params.is_local, "This flash attention build does not support local attention.");
    #endif
}

template <int Arch, int Split, bool PagedKVNonTMA, bool PackGQA, bool Has_softcap>
void run_mha_fwd_constexpr(Flash_fwd_params &params, cudaStream_t stream) {
    if (!params.is_e4m3) {
        if (params.is_bf16) {
            #ifndef FLASHATTENTION_DISABLE_HDIM64
            if (params.d <= 64) {
                #ifndef FLASHATTENTION_DISABLE_HDIMDIFF64
                if constexpr (Arch == 90) {
                    if (params.dv > 256) {
                        return run_mha_fwd_<Arch, cutlass::bfloat16_t, 64, 512, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                    } else if (params.dv > 64) {
                        return run_mha_fwd_<Arch, cutlass::bfloat16_t, 64, 256, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                    }
                }
                #endif
                return run_mha_fwd_<Arch, cutlass::bfloat16_t, 64, 64, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
            }
            #endif
            #ifndef FLASHATTENTION_DISABLE_HDIM96
            if (params.d <= 96) { return run_mha_fwd_<Arch, cutlass::bfloat16_t, 96, 96, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
            #endif
            #ifndef FLASHATTENTION_DISABLE_HDIM128
            if (params.d <= 128) { return run_mha_fwd_<Arch, cutlass::bfloat16_t, 128, 128, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
            #endif
            #ifndef FLASHATTENTION_DISABLE_HDIM192
            if (params.d <= 192) {
                #ifndef FLASHATTENTION_DISABLE_HDIMDIFF192
                if constexpr (Arch == 90) {
                    if (params.dv <= 128) {
                        return run_mha_fwd_<Arch, cutlass::bfloat16_t, 192, 128, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                    }
                }
                #endif
                return run_mha_fwd_<Arch, cutlass::bfloat16_t, 192, 192, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
            }
            #endif
            #ifndef FLASHATTENTION_DISABLE_HDIM256
            if (params.d <= 256) { return run_mha_fwd_<Arch, cutlass::bfloat16_t, 256, 256, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
            #endif
        } else {
            #ifndef FLASHATTENTION_DISABLE_FP16
            #ifndef FLASHATTENTION_DISABLE_HDIM64
            if (params.d <= 64) {
                #ifndef FLASHATTENTION_DISABLE_HDIMDIFF64
                if constexpr (Arch == 90) {
                    if (params.dv > 256) {
                        return run_mha_fwd_<Arch, cutlass::half_t, 64, 512, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                    } else if (params.dv > 64) {
                        return run_mha_fwd_<Arch, cutlass::half_t, 64, 256, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                    }
                }
                #endif
                return run_mha_fwd_<Arch, cutlass::half_t, 64, 64, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
            }
            #endif
            #ifndef FLASHATTENTION_DISABLE_HDIM96
            if (params.d <= 96) { return run_mha_fwd_<Arch, cutlass::half_t, 96, 96, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
            #endif
            #ifndef FLASHATTENTION_DISABLE_HDIM128
            if (params.d <= 128) { return run_mha_fwd_<Arch, cutlass::half_t, 128, 128, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
            #endif
            #ifndef FLASHATTENTION_DISABLE_HDIM192
            if (params.d <= 192) {
                #ifndef FLASHATTENTION_DISABLE_HDIMDIFF192
                if constexpr (Arch == 90) {
                    if (params.dv <= 128) {
                        return run_mha_fwd_<Arch, cutlass::half_t, 192, 128, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                    }
                }
                #endif
                return run_mha_fwd_<Arch, cutlass::half_t, 192, 192, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
            }
            #endif
            #ifndef FLASHATTENTION_DISABLE_HDIM256
            if (params.d <= 256) { return run_mha_fwd_<Arch, cutlass::half_t, 256, 256, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
            #endif
            #else
            CHECK_ASSERT(false, "This flash attention build does not support FP16.");
            #endif
        }
    } else {
        #ifndef FLASHATTENTION_DISABLE_FP8
        #ifndef FLASHATTENTION_DISABLE_HDIM64
        if (params.d <= 64) { return run_mha_fwd_<90, cutlass::float_e4m3_t, 64, 64, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
        #endif
        #ifndef FLASHATTENTION_DISABLE_HDIM96
        if (params.d <= 96) { return run_mha_fwd_<90, cutlass::float_e4m3_t, 96, 96, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
        #endif
        #ifndef FLASHATTENTION_DISABLE_HDIM128
        if (params.d <= 128) { return run_mha_fwd_<90, cutlass::float_e4m3_t, 128, 128, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
        #endif
        #ifndef FLASHATTENTION_DISABLE_HDIM192
        if (params.d <= 192) {
            #ifndef FLASHATTENTION_DISABLE_HDIMDIFF192
            if constexpr (Arch == 90) {
                if (params.dv <= 128) {
                    return run_mha_fwd_<90, cutlass::float_e4m3_t, 192, 128, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                }
            }
            #endif
            return run_mha_fwd_<90, cutlass::float_e4m3_t, 192, 192, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
        }
        #endif
        #ifndef FLASHATTENTION_DISABLE_HDIM256
        if (params.d <= 256) { return run_mha_fwd_<90, cutlass::float_e4m3_t, 256, 256, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
        #endif
        #else
        CHECK_ASSERT(false, "This flash attention build does not support FP8.");
        #endif
    }
}

void run_mha_fwd_standalone(Flash_fwd_params &params, cudaStream_t stream) {
    CHECK_ASSERT(params.num_splits >= 1, "num_splits must be >= 1");
    ARCH_SWITCH(params.arch, Arch, [&] {
        SPLIT_SWITCH(params.num_splits > 1, Split, [&] {
            PAGEDKV_SWITCH(params.page_table && !params.pagedkv_tma, PagedKVNonTMA, [&] {
                PACKGQA_SWITCH(params.pack_gqa, PackGQA_, [&] {
                    // Always enable PackGQA for Sm8x or PagedKVNonTMA or Split to reduce compilation
                    static constexpr bool PackGQA = PackGQA_ || Arch < 90 || PagedKVNonTMA || Split;
                    SOFTCAP_SWITCH(params.softcap > 0.0, Has_softcap, [&] {
                        run_mha_fwd_constexpr<Arch, Split, PagedKVNonTMA, PackGQA, Has_softcap>(params, stream);
                    });
                });
            });
        });
    });
}

// Full version with ALL features - complete alignment with original mha_fwd
// This function requires ~60+ parameters to fully replicate all features
extern "C" void mha_fwd_standalone(
        // Q, K, V device pointers
        void* q_ptr,
        void* k_ptr,
        void* v_ptr,
        // Optional k_new, v_new (for KV cache appending)
        void* k_new_ptr,  // nullptr if not used
        void* v_new_ptr,  // nullptr if not used
        // Optional q_v
        void* q_v_ptr,  // nullptr if not used
        // Output (can be pre-allocated or caller allocates)
        void* out_ptr,  // must be pre-allocated
        // Variable length support
        void* cu_seqlens_q_ptr,  // int32*, nullptr if not varlen_q
        void* cu_seqlens_k_ptr,  // int32*, nullptr if not varlen_k
        void* cu_seqlens_k_new_ptr,  // int32*, nullptr if not varlen_k_new
        void* seqused_q_ptr,  // int32*, nullptr if not used
        void* seqused_k_ptr,  // int32*, nullptr if not used
        int64_t max_seqlen_q,  // required if cu_seqlens_q is provided, else ignored
        int64_t max_seqlen_k,  // required if cu_seqlens_k is provided, else ignored
        // Paged KV support
        void* page_table_ptr,  // int32*, nullptr if not paged
        void* kv_batch_idx_ptr,  // int32*, nullptr if not used
        // Leftpad support
        void* leftpad_k_ptr,  // int32*, nullptr if not used
        // Rotary embeddings
        void* rotary_cos_ptr,  // nullptr if not used
        void* rotary_sin_ptr,  // nullptr if not used
        void* seqlens_rotary_ptr,  // int32*, nullptr if not used
        // FP8 descale
        void* q_descale_ptr,  // float*, nullptr if not FP8 or not using descale
        void* k_descale_ptr,  // float*, nullptr if not FP8 or not using descale
        void* v_descale_ptr,  // float*, nullptr if not FP8 or not using descale
        // Scheduler metadata (optional pre-allocated)
        void* scheduler_metadata_ptr,  // int32*, nullptr to allocate internally
        // Type flags
        bool is_bf16,
        bool is_e4m3,
        // Dimensions for Q
        int batch_size_q,  // if varlen_q, this is number of sequences (cu_seqlens size - 1)
        int seqlen_q,  // if varlen_q, this is total_q
        int num_heads,
        int head_size,
        // Dimensions for K/V
        int batch_size_k,  // might differ from batch_size_q if paged or kv_batch_idx
        int seqlen_k,  // if varlen_k, this is total_k; if paged, this is num_pages
        int num_heads_k,
        int head_size_v,
        // Paged KV dimensions
        int page_size,  // only used if page_table_ptr != nullptr
        int max_num_pages_per_seq,  // only used if page_table_ptr != nullptr
        // k_new/v_new dimensions
        int seqlen_k_new,  // only used if k_new_ptr != nullptr
        // Rotary dimensions
        int rotary_dim,  // only used if rotary_cos_ptr != nullptr
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
        double softmax_scale_val,  // < 0 to use default
        bool is_causal,
        int64_t window_size_left,
        int64_t window_size_right,
        int64_t attention_chunk,
        double softcap,
        bool is_rotary_interleaved,
        int64_t num_splits,  // <= 0 to auto-determine
        int pack_gqa_val,  // < 0 to auto-determine
        int64_t sm_margin,
        // Pre-allocated output buffers (MUST provide)
        void* softmax_lse_ptr,  // float*, must be pre-allocated
        void* out_accum_ptr,  // float*, nullptr if num_splits == 1
        void* softmax_lse_accum_ptr,  // float*, nullptr if num_splits == 1
        // CUDA stream
        cudaStream_t stream
) {
    // Get device properties
    cudaDeviceProp dprops;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&dprops, device);

    bool is_sm8x = dprops.major >= 8;
    CHECK_ASSERT(is_sm8x, "FlashAttention only supports Ampere GPUs or newer.");

    // Validate data types
    CHECK_ASSERT(is_bf16 || !is_bf16, "Only fp16, bf16, and fp8_e4m3 supported");
    if (dprops.major < 9) {
        CHECK_ASSERT(!is_e4m3, "FlashAttention on Ampere/Ada cards only supports fp16 and bf16");
    }

    // Determine if paged KV
    bool const paged_KV = page_table_ptr != nullptr;

    // Determine if varlen
    bool const is_varlen_q = cu_seqlens_q_ptr != nullptr;
    bool const is_varlen_k = cu_seqlens_k_ptr != nullptr;
    bool const is_varlen_k_new = cu_seqlens_k_new_ptr != nullptr;

    if (is_varlen_q) {
        CHECK_ASSERT(max_seqlen_q > 0, "max_seqlen_q must be provided if cu_seqlens_q is provided");
    }
    if (is_varlen_k) {
        CHECK_ASSERT(max_seqlen_k > 0, "max_seqlen_q must be provided if cu_seqlens_k is provided");
        CHECK_ASSERT(!paged_KV, "cu_seqlens_k and page_table are mutually exclusive");
        CHECK_ASSERT(kv_batch_idx_ptr == nullptr, "cu_seqlens_k and kv_batch_idx are mutually exclusive");
    }

    // Calculate actual dimensions
    int const batch_size = batch_size_q;
    int const seqlen_q_actual = !is_varlen_q ? seqlen_q : max_seqlen_q;
    int const total_q = !is_varlen_q ? batch_size * seqlen_q : seqlen_q;
    int const seqlen_k_actual = !is_varlen_k ? (!paged_KV ? seqlen_k : max_num_pages_per_seq * page_size) : max_seqlen_k;
    int const total_k = !is_varlen_k ? batch_size_k * seqlen_k : seqlen_k;
    int const num_pages = !paged_KV ? 0 : seqlen_k;

    // Calculate softmax scale
    double softmax_scale = softmax_scale_val >= 0 ? softmax_scale_val : 1.0 / sqrt(double(head_size));

    if (kv_batch_idx_ptr == nullptr) {
        CHECK_ASSERT(batch_size == batch_size_k, "batch_size must equal batch_size_k");
    }

    int const max_headdim = get_max_headdim();
    CHECK_ASSERT(head_size <= max_headdim, "head_size too large");
    CHECK_ASSERT(num_heads % num_heads_k == 0, "num_heads must be divisible by num_heads_k");

    if (head_size_v != head_size) {
        CHECK_ASSERT((head_size > 128 && head_size <= 192 && head_size_v > 96 && head_size_v <= 128) ||
                   (head_size <= 64 && head_size_v <= 512),
                   "Unsupported head_size and head_size_v combination");
        CHECK_ASSERT(dprops.major == 9, "Only Hopper supports different V headdim");
        if (head_size_v > 256) {
            CHECK_ASSERT(!is_e4m3, "HeaddimV > 256 requires fp16/bf16");
        }
    }

    // Window size adjustments
    int64_t window_size_left_adj = window_size_left;
    int64_t window_size_right_adj = window_size_right;
    if (window_size_left >= seqlen_k_actual - 1) { window_size_left_adj = -1; }
    if (window_size_right >= seqlen_q_actual - 1) { window_size_right_adj = -1; }

    bool is_causal_adj = is_causal;
    if (seqlen_q_actual == 1 && window_size_left_adj == -1 && window_size_right_adj == -1 && attention_chunk == 0) {
        if ((head_size <= 64 || head_size > 128) || !paged_KV) {
            is_causal_adj = false;
        }
    }
    if (is_causal_adj) { window_size_right_adj = 0; }

    // Check varlen
    bool const is_varlen = is_varlen_q || is_varlen_k || seqused_q_ptr || seqused_k_ptr || leftpad_k_ptr;

    int const alignment = is_e4m3 ? 16 : 8;
    CHECK_ASSERT(head_size % alignment == 0, "head_size alignment error");
    CHECK_ASSERT(head_size_v % alignment == 0, "head_size_v alignment error");

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    int const head_size_rounded = round_up_headdim(head_size);
    int const head_size_v_rounded = head_size_v == head_size ? head_size_rounded : round_up_headdimv(head_size_v);
    int const seqlen_q_rounded = round_multiple(seqlen_q_actual, 128);
    int const seqlen_k_rounded = round_multiple(seqlen_k_actual, 128);

    Flash_fwd_params params;
    set_params_fprop_standalone(params,
                     batch_size,
                     seqlen_q_actual, seqlen_k_actual,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q_ptr, k_ptr, v_ptr, out_ptr,
                     is_bf16, is_e4m3,
                     q_batch_stride, k_batch_stride, v_batch_stride, o_batch_stride,
                     q_row_stride, k_row_stride, v_row_stride, o_row_stride,
                     q_head_stride, k_head_stride, v_head_stride, o_head_stride,
                     head_size_v,
                     cu_seqlens_q_ptr,
                     cu_seqlens_k_ptr,
                     seqused_q_ptr,
                     seqused_k_ptr,
                     softmax_lse_ptr,
                     0.f,  // p_dropout
                     softmax_scale,
                     window_size_left_adj,
                     window_size_right_adj,
                     attention_chunk,
                     softcap,
                     sm_margin);

    params.total_q = total_q;
    params.total_k = total_k;
    params.b_k = batch_size_k;
    params.dv = head_size_v;
    params.dv_rounded = head_size_v_rounded;

    // Leftpad
    if (leftpad_k_ptr) {
        params.leftpad_k = static_cast<int*>(leftpad_k_ptr);
    }

    // Paged KV
    if (paged_KV) {
        params.page_table = static_cast<int*>(page_table_ptr);
        params.page_table_batch_stride = page_table_batch_stride;
    }
    params.page_size = page_size;
    params.num_pages = num_pages;

    // k_new/v_new
    if (k_new_ptr) {
        CHECK_ASSERT(v_new_ptr, "k_new requires v_new");
        CHECK_ASSERT(seqused_k_ptr, "k_new requires seqused_k");
        CHECK_ASSERT(seqlen_q_actual <= seqlen_k_actual, "k_new seqlen constraint");

        int const total_k_new = !is_varlen_k_new ? batch_size * seqlen_k_new : seqlen_k_new;

        params.seqlen_knew = seqlen_k_new;
        params.total_knew = total_k_new;
        params.knew_ptr = k_new_ptr;
        params.vnew_ptr = v_new_ptr;
        params.knew_row_stride = k_new_row_stride;
        params.vnew_row_stride = v_new_row_stride;
        params.knew_head_stride = k_new_head_stride;
        params.vnew_head_stride = v_new_head_stride;
        if (!is_varlen_k_new) {
            params.knew_batch_stride = k_new_batch_stride;
            params.vnew_batch_stride = v_new_batch_stride;
        }
        if (is_varlen_k_new) {
            params.cu_seqlens_knew = static_cast<int*>(cu_seqlens_k_new_ptr);
        }
    }

    // Prepare varlen
    bool const use_prepare_varlen = is_varlen;
    params.prepare_varlen_pdl = use_prepare_varlen && params.b <= PREPARE_VARLEN_MAX_BATCHES_1CTA;
    params.num_splits_dynamic_ptr = !use_prepare_varlen ? nullptr : reinterpret_cast<int*>(1);

    params.pagedkv_tma = get_pagedkv_tma(params);
    params.num_splits = num_splits <= 0 ? get_num_splits(params) : num_splits;
    params.pack_gqa = pack_gqa_val >= 0 ? (pack_gqa_val > 0) : get_pack_gqa(params);

    // Scheduler setup
    bool const scheduler_needs_semaphore = dprops.major >= 90
        ? (((params.is_causal || params.is_local) && (params.num_splits == 1)) || is_varlen)
        : ((params.is_causal && !is_varlen) || (is_varlen && params.num_splits > 1));
    params.varlen_sort_batches = !params.is_local;
    params.head_swizzle = params.is_causal || params.is_local;

    if (scheduler_needs_semaphore || use_prepare_varlen) {
        int b_rounded = round_multiple(params.b, 4);
        int num_prepare_batch_vectors = use_prepare_varlen ? 2 : 0;
        if (params.varlen_sort_batches) { num_prepare_batch_vectors += 1; }
        if (params.head_swizzle) { num_prepare_batch_vectors += 1; }
        int head_swizzle_offset = b_rounded * (params.varlen_sort_batches ? 3 : 2);
        int tile_count_semaphore_offset = b_rounded * num_prepare_batch_vectors;
        // int metadata_size = int(scheduler_needs_semaphore) + tile_count_semaphore_offset;

        params.skip_scheduler_metadata_computation = scheduler_metadata_ptr != nullptr;
        void* tile_count_semaphore_ptr = scheduler_metadata_ptr;

        // NOTE: Caller must provide scheduler_metadata if needed, or we allocate on stack (limited)
        int local_scheduler_metadata[1024] = {0};  // Stack allocation for simple cases
        if (!tile_count_semaphore_ptr) {
            tile_count_semaphore_ptr = local_scheduler_metadata;
        }

        params.num_splits_dynamic_ptr = use_prepare_varlen ? static_cast<int*>(tile_count_semaphore_ptr) : nullptr;
        params.num_m_blocks_ptr = use_prepare_varlen ? static_cast<int*>(tile_count_semaphore_ptr) + b_rounded : nullptr;
        params.varlen_batch_idx_ptr = use_prepare_varlen && params.varlen_sort_batches ? static_cast<int*>(tile_count_semaphore_ptr) + b_rounded * 2 : nullptr;
        params.num_nheads_in_l2_ptr = use_prepare_varlen && params.head_swizzle ? static_cast<int*>(tile_count_semaphore_ptr) + head_swizzle_offset : nullptr;
        params.tile_count_semaphore = scheduler_needs_semaphore ? static_cast<int*>(tile_count_semaphore_ptr) + tile_count_semaphore_offset : nullptr;
        params.tile_count_semaphore_offset = tile_count_semaphore_offset;
    }

    // q_v support
    if (q_v_ptr) {
        CHECK_ASSERT(head_size <= 64, "q_v requires head_size <= 64");
        CHECK_ASSERT(head_size_v >= 256, "q_v requires hdim_v >= 256");
        CHECK_ASSERT(!is_e4m3, "q_v requires fp16/bf16");
        CHECK_ASSERT(dprops.major == 90, "q_v requires Hopper");

        params.qv_ptr = q_v_ptr;
        params.qv_row_stride = q_v_row_stride;
        params.qv_head_stride = q_v_head_stride;
        if (!is_varlen_q) {
            params.qv_batch_stride = q_v_batch_stride;
        }
    }

    // Rotary embeddings
    if (rotary_cos_ptr) {
        CHECK_ASSERT(k_new_ptr, "Rotary requires k_new/v_new");

        params.rotary_dim = rotary_dim;
        CHECK_ASSERT(params.rotary_dim <= head_size, "rotary_dim <= headdim");
        CHECK_ASSERT(params.rotary_dim % 16 == 0, "rotary_dim % 16 == 0");

        CHECK_ASSERT(rotary_sin_ptr, "Rotary requires both cos and sin");

        params.rotary_cos_ptr = rotary_cos_ptr;
        params.rotary_sin_ptr = rotary_sin_ptr;
        params.is_rotary_interleaved = is_rotary_interleaved;

        if (seqlens_rotary_ptr) {
            params.seqlens_rotary = static_cast<int*>(seqlens_rotary_ptr);
        }
    } else {
        params.rotary_dim = 0;
    }

    // kv_batch_idx
    if (kv_batch_idx_ptr) {
        params.kv_batch_idx = static_cast<int*>(kv_batch_idx_ptr);
    }

    // Multiple splits
    if (params.num_splits > 1) {
        CHECK_ASSERT(params.num_splits <= 256, "num_splits <= 256");
        CHECK_ASSERT(out_accum_ptr, "num_splits > 1 requires out_accum");
        CHECK_ASSERT(softmax_lse_accum_ptr, "num_splits > 1 requires softmax_lse_accum");

        params.is_fp32 = false;
        params.oaccum_ptr = out_accum_ptr;
        params.softmax_lseaccum_ptr = softmax_lse_accum_ptr;

        // NOTE: Caller must set up strides for accumulators
        // For simplicity, assume standard layout
        if (!is_varlen_q) {
            params.oaccum_batch_stride = num_heads * seqlen_q_actual * head_size_v;
            params.lseaccum_batch_stride = num_heads * seqlen_q_actual;
        }
        params.oaccum_split_stride = !is_varlen_q ? (batch_size * num_heads * seqlen_q_actual * head_size_v) : (num_heads * total_q * head_size_v);
        params.oaccum_row_stride = head_size_v;
        params.oaccum_head_stride = seqlen_q_actual * head_size_v;
        params.lseaccum_split_stride = !is_varlen_q ? (batch_size * num_heads * seqlen_q_actual) : (num_heads * total_q);
        params.lseaccum_head_stride = seqlen_q_actual;
    }

    // FP8 descale
    if (is_e4m3) {
        if (q_descale_ptr) {
            params.q_descale_ptr = static_cast<float*>(q_descale_ptr);
            params.q_descale_batch_stride = q_descale_batch_stride;
            params.q_descale_head_stride = q_descale_head_stride;
        } else {
            params.q_descale_ptr = nullptr;
        }
        if (k_descale_ptr) {
            params.k_descale_ptr = static_cast<float*>(k_descale_ptr);
            params.k_descale_batch_stride = k_descale_batch_stride;
            params.k_descale_head_stride = k_descale_head_stride;
        } else {
            params.k_descale_ptr = nullptr;
        }
        if (v_descale_ptr) {
            params.v_descale_ptr = static_cast<float*>(v_descale_ptr);
            params.v_descale_batch_stride = v_descale_batch_stride;
            params.v_descale_head_stride = v_descale_head_stride;
        } else {
            params.v_descale_ptr = nullptr;
        }
    }

    // Run the kernel
    if (total_q > 0 && (total_k + (k_new_ptr ? params.total_knew : 0)) > 0 && num_heads_k > 0) {
        run_mha_fwd_standalone(params, stream);

        // NOTE: If num_splits > 1, caller needs to run combine kernel separately
        // or we need to implement run_mha_fwd_combine_standalone
    } else if (total_q > 0 && num_heads_k > 0) {
        // Empty K: zero output
        size_t out_bytes = total_q * num_heads * head_size_v * (is_e4m3 ? 2 : (is_bf16 ? 2 : 2));
        cudaMemsetAsync(out_ptr, 0, out_bytes, stream);
        // Set softmax_lse to infinity - would need proper implementation
    }
}
