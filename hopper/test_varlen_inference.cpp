/******************************************************************************
 * Test program for FlashAttnVarlenFunc inference with standalone API
 * Uses random initialized data and command-line arguments
 ******************************************************************************/

#include <cuda_runtime.h>
#include <curand.h>
#include <iostream>
#include <vector>
#include <random>
#include <cstring>
#include <cmath>

// Forward declaration of mha_fwd_standalone
extern "C" void mha_fwd_standalone(
        void* q_ptr, void* k_ptr, void* v_ptr,
        void* k_new_ptr, void* v_new_ptr, void* q_v_ptr, void* out_ptr,
        void* cu_seqlens_q_ptr, void* cu_seqlens_k_ptr, void* cu_seqlens_k_new_ptr,
        void* seqused_q_ptr, void* seqused_k_ptr,
        int64_t max_seqlen_q, int64_t max_seqlen_k,
        void* page_table_ptr, void* kv_batch_idx_ptr, void* leftpad_k_ptr,
        void* rotary_cos_ptr, void* rotary_sin_ptr, void* seqlens_rotary_ptr,
        void* q_descale_ptr, void* k_descale_ptr, void* v_descale_ptr,
        void* scheduler_metadata_ptr,
        bool is_bf16, bool is_e4m3,
        int batch_size_q, int seqlen_q, int num_heads, int head_size,
        int batch_size_k, int seqlen_k, int num_heads_k, int head_size_v,
        int page_size, int max_num_pages_per_seq, int seqlen_k_new, int rotary_dim,
        int64_t q_batch_stride, int64_t q_row_stride, int64_t q_head_stride,
        int64_t k_batch_stride, int64_t k_row_stride, int64_t k_head_stride,
        int64_t v_batch_stride, int64_t v_row_stride, int64_t v_head_stride,
        int64_t o_batch_stride, int64_t o_row_stride, int64_t o_head_stride,
        int64_t k_new_batch_stride, int64_t k_new_row_stride, int64_t k_new_head_stride,
        int64_t v_new_batch_stride, int64_t v_new_row_stride, int64_t v_new_head_stride,
        int64_t q_v_batch_stride, int64_t q_v_row_stride, int64_t q_v_head_stride,
        int64_t page_table_batch_stride,
        int64_t q_descale_batch_stride, int64_t q_descale_head_stride,
        int64_t k_descale_batch_stride, int64_t k_descale_head_stride,
        int64_t v_descale_batch_stride, int64_t v_descale_head_stride,
        double softmax_scale_val, bool is_causal,
        int64_t window_size_left, int64_t window_size_right,
        int64_t attention_chunk, double softcap, bool is_rotary_interleaved,
        int64_t num_splits, int pack_gqa_val, int64_t sm_margin,
        void* softmax_lse_ptr, void* out_accum_ptr, void* softmax_lse_accum_ptr,
        cudaStream_t stream
);

// Helper to initialize random data on GPU using cuRAND
void init_random_data_gpu(void* d_ptr, size_t num_elements, bool is_fp16, bool is_bf16, bool is_fp8) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

    if (is_fp8 || is_fp16 || is_bf16) {
        // Generate random floats then convert
        float* temp;
        cudaMalloc(&temp, num_elements * sizeof(float));
        curandGenerateUniform(gen, temp, num_elements);

        // Scale to reasonable range for attention (0, 1) -> (-0.5, 0.5)
        // Simple kernel would be needed here, for now just copy
        cudaMemcpy(d_ptr, temp, num_elements * 2, cudaMemcpyDeviceToDevice); // FP16/BF16 are 2 bytes
        cudaFree(temp);
    }

    curandDestroyGenerator(gen);
}

int main(int argc, char** argv) {
    // Default parameters (can be overridden by command line arguments)
    // NOTE: This build only supports FP16 hdim=128 and FP8 hdim=96
    int batch_size = 2;
    int max_seqlen_q = 512;
    int max_seqlen_k = 512;
    int num_heads = 8;
    int num_heads_k = 8;  // GQA: can be different from num_heads
    int head_size = 128;  // Default to 128 for FP16 support
    int head_size_v = 128;
    bool is_causal = false;
    bool is_bf16 = false;  // Default to FP16 (BF16 not supported in this build)
    bool is_e4m3 = false;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--batch_size") == 0 && i + 1 < argc) {
            batch_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--max_seqlen_q") == 0 && i + 1 < argc) {
            max_seqlen_q = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--max_seqlen_k") == 0 && i + 1 < argc) {
            max_seqlen_k = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--num_heads") == 0 && i + 1 < argc) {
            num_heads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--num_heads_k") == 0 && i + 1 < argc) {
            num_heads_k = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--head_size") == 0 && i + 1 < argc) {
            head_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--causal") == 0) {
            is_causal = true;
        } else if (strcmp(argv[i], "--fp16") == 0) {
            is_bf16 = false;
            is_e4m3 = false;
        } else if (strcmp(argv[i], "--bf16") == 0) {
            is_bf16 = true;
            is_e4m3 = false;
        } else if (strcmp(argv[i], "--fp8") == 0) {
            is_bf16 = false;
            is_e4m3 = true;
        } else if (strcmp(argv[i], "--help") == 0) {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "\nNOTE: This build only supports:\n"
                      << "  - FP16 with head_dim <= 128 (default)\n"
                      << "  - FP8 E4M3 with head_dim <= 96\n"
                      << "  - BF16 is NOT supported\n"
                      << "\nOptions:\n"
                      << "  --batch_size N       Batch size (default: 2)\n"
                      << "  --max_seqlen_q N     Max sequence length for Q (default: 512)\n"
                      << "  --max_seqlen_k N     Max sequence length for K (default: 512)\n"
                      << "  --num_heads N        Number of query heads (default: 8)\n"
                      << "  --num_heads_k N      Number of key/value heads (default: 8)\n"
                      << "  --head_size N        Head dimension (default: 128 for FP16, use 96 for FP8)\n"
                      << "  --causal             Use causal attention (default: false)\n"
                      << "  --fp16               Use FP16 (default)\n"
                      << "  --fp8                Use FP8 E4M3 (requires head_size <= 96)\n"
                      << "  --help               Show this help message\n";
            return 0;
        }
    }

    head_size_v = head_size;  // For simplicity, same as head_size

    std::cout << "FlashAttention Varlen Inference Test\n"
              << "====================================\n"
              << "Batch size: " << batch_size << "\n"
              << "Max seqlen Q: " << max_seqlen_q << "\n"
              << "Max seqlen K: " << max_seqlen_k << "\n"
              << "Num heads: " << num_heads << "\n"
              << "Num heads K: " << num_heads_k << "\n"
              << "Head size: " << head_size << "\n"
              << "Data type: " << (is_e4m3 ? "FP8" : (is_bf16 ? "BF16" : "FP16")) << "\n"
              << "Causal: " << (is_causal ? "Yes" : "No") << "\n\n";

    // Use fixed sequence lengths equal to max_seqlen
    // This ensures consistent benchmarking with the specified parameters
    std::vector<int> seqlens_q(batch_size);
    std::vector<int> seqlens_k(batch_size);
    int total_q = 0, total_k = 0;

    std::cout << "Sequence lengths (all fixed to max_seqlen):\n";
    for (int i = 0; i < batch_size; i++) {
        seqlens_q[i] = max_seqlen_q;
        seqlens_k[i] = max_seqlen_k;
        total_q += seqlens_q[i];
        total_k += seqlens_k[i];
        std::cout << "  Batch " << i << ": Q=" << seqlens_q[i] << ", K=" << seqlens_k[i] << "\n";
    }

    // Create cumulative sequence lengths (cu_seqlens)
    std::vector<int> cu_seqlens_q(batch_size + 1);
    std::vector<int> cu_seqlens_k(batch_size + 1);
    cu_seqlens_q[0] = 0;
    cu_seqlens_k[0] = 0;
    for (int i = 0; i < batch_size; i++) {
        cu_seqlens_q[i + 1] = cu_seqlens_q[i] + seqlens_q[i];
        cu_seqlens_k[i + 1] = cu_seqlens_k[i] + seqlens_k[i];
    }

    std::cout << "\nTotal Q tokens: " << total_q << "\n";
    std::cout << "Total K tokens: " << total_k << "\n\n";

    // Allocate device memory
    size_t dtype_size = is_e4m3 ? 1 : 2;  // FP8=1 byte, FP16/BF16=2 bytes

    // Q: (total_q, num_heads, head_size)
    void *d_q, *d_k, *d_v, *d_out;
    void *d_cu_seqlens_q, *d_cu_seqlens_k;
    void *d_softmax_lse;
    void *d_scheduler_metadata;

    cudaMalloc(&d_q, total_q * num_heads * head_size * dtype_size);
    cudaMalloc(&d_k, total_k * num_heads_k * head_size * dtype_size);
    cudaMalloc(&d_v, total_k * num_heads_k * head_size_v * dtype_size);
    cudaMalloc(&d_out, total_q * num_heads * head_size_v * dtype_size);

    cudaMalloc(&d_cu_seqlens_q, (batch_size + 1) * sizeof(int));
    cudaMalloc(&d_cu_seqlens_k, (batch_size + 1) * sizeof(int));

    // softmax_lse: (num_heads, total_q) for varlen
    cudaMalloc(&d_softmax_lse, num_heads * total_q * sizeof(float));

    // Scheduler metadata for varlen attention
    // Size calculation: b_rounded * num_prepare_batch_vectors + tile_count_semaphore_offset
    // Allocate generous buffer: 1024 int32 values
    cudaMalloc(&d_scheduler_metadata, 1024 * sizeof(int));
    cudaMemset(d_scheduler_metadata, 0, 1024 * sizeof(int));

    // Copy cu_seqlens to device
    cudaMemcpy(d_cu_seqlens_q, cu_seqlens_q.data(), (batch_size + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cu_seqlens_k, cu_seqlens_k.data(), (batch_size + 1) * sizeof(int), cudaMemcpyHostToDevice);

    // Initialize random data
    std::cout << "Initializing random data...\n";
    init_random_data_gpu(d_q, total_q * num_heads * head_size, !is_bf16 && !is_e4m3, is_bf16, is_e4m3);
    init_random_data_gpu(d_k, total_k * num_heads_k * head_size, !is_bf16 && !is_e4m3, is_bf16, is_e4m3);
    init_random_data_gpu(d_v, total_k * num_heads_k * head_size_v, !is_bf16 && !is_e4m3, is_bf16, is_e4m3);

    // Calculate strides (for varlen, batch_stride is not used)
    int64_t q_row_stride = num_heads * head_size;
    int64_t q_head_stride = head_size;
    int64_t k_row_stride = num_heads_k * head_size;
    int64_t k_head_stride = head_size;
    int64_t v_row_stride = num_heads_k * head_size_v;
    int64_t v_head_stride = head_size_v;
    int64_t o_row_stride = num_heads * head_size_v;
    int64_t o_head_stride = head_size_v;

    // Create CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Single run
    std::cout << "Running attention kernel...\n";

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, stream);
    mha_fwd_standalone(
        d_q, d_k, d_v,
        nullptr, nullptr, nullptr,  // k_new, v_new, q_v
        d_out,
        d_cu_seqlens_q, d_cu_seqlens_k, nullptr,  // cu_seqlens
        nullptr, nullptr,  // seqused_q, seqused_k
        max_seqlen_q, max_seqlen_k,
        nullptr, nullptr, nullptr,  // page_table, kv_batch_idx, leftpad_k
        nullptr, nullptr, nullptr,  // rotary
        nullptr, nullptr, nullptr,  // FP8 descale
        d_scheduler_metadata,  // scheduler_metadata
        is_bf16, is_e4m3,
        batch_size, total_q, num_heads, head_size,  // Q dims (seqlen_q is total_q for varlen)
        batch_size, total_k, num_heads_k, head_size_v,  // K dims (seqlen_k is total_k for varlen)
        0, 0, 0, 0,  // paged KV dims, rotary dim
        0, q_row_stride, q_head_stride,  // Q strides (batch_stride ignored for varlen)
        0, k_row_stride, k_head_stride,  // K strides
        0, v_row_stride, v_head_stride,  // V strides
        0, o_row_stride, o_head_stride,  // O strides
        0, 0, 0, 0, 0, 0,  // k_new, v_new strides
        0, 0, 0,  // q_v strides
        0,  // page_table stride
        0, 0, 0, 0, 0, 0,  // FP8 descale strides
        -1.0,  // softmax_scale (auto)
        is_causal,
        -1, -1,  // window_size
        0,  // attention_chunk
        0.0,  // softcap
        false,  // is_rotary_interleaved
        -1,  // num_splits (auto)
        -1,  // pack_gqa (auto)
        0,  // sm_margin
        d_softmax_lse,
        nullptr, nullptr,  // out_accum, softmax_lse_accum (for splits)
        stream
    );
    cudaEventRecord(stop, stream);
    cudaStreamSynchronize(stream);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Calculate FLOPs and bandwidth
    // Attention FLOPs ≈ 4 * total_q * total_k * head_size * num_heads
    double flops = 4.0 * total_q * total_k * head_size * num_heads;
    double tflops = flops / (milliseconds * 1e9);  // TFLOP/s

    std::cout << "\nResults:\n"
              << "========\n"
              << "Execution time: " << milliseconds << " ms\n"
              << "Throughput: " << tflops << " TFLOP/s\n";

    // Cleanup
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_out);
    cudaFree(d_cu_seqlens_q);
    cudaFree(d_cu_seqlens_k);
    cudaFree(d_softmax_lse);
    cudaFree(d_scheduler_metadata);
    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::cout << "\nTest completed successfully!\n";

    return 0;
}
