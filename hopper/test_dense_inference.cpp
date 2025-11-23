#include <iostream>
#include <vector>
#include <cstring>
#include <cuda_runtime.h>
#include <curand.h>
#include "flash_api_standalone.h"

// Helper function to initialize random data on GPU
void init_random_data_gpu(void* d_ptr, size_t num_elements, bool is_fp16, bool is_bf16, bool is_fp8) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

    if (is_fp8) {
        // For FP8, generate as FP32 then convert
        float* temp;
        cudaMalloc(&temp, num_elements * sizeof(float));
        curandGenerateUniform(gen, temp, num_elements);

        // Simple FP32 to FP8 conversion kernel would go here
        // For now, just copy (note: this is incorrect but for testing structure)
        cudaMemcpy(d_ptr, temp, num_elements, cudaMemcpyDeviceToDevice);
        cudaFree(temp);
    } else {
        // For FP16/BF16, generate as FP32 then convert
        float* temp;
        cudaMalloc(&temp, num_elements * sizeof(float));
        curandGenerateUniform(gen, temp, num_elements);

        // Copy to half precision (simplified, assumes runtime conversion)
        cudaMemcpy(d_ptr, temp, num_elements * sizeof(uint16_t), cudaMemcpyDeviceToDevice);
        cudaFree(temp);
    }

    curandDestroyGenerator(gen);
}

int main(int argc, char** argv) {
    // Default parameters - DENSE (non-varlen) format
    int batch_size = 2;
    int seqlen_q = 512;
    int seqlen_k = 512;
    int num_heads = 8;
    int num_heads_k = 8;
    int head_size = 128;
    int head_size_v = 128;
    bool is_causal = false;
    bool is_bf16 = false;
    bool is_e4m3 = false;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--batch_size") == 0 && i + 1 < argc) {
            batch_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seqlen_q") == 0 && i + 1 < argc) {
            seqlen_q = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seqlen_k") == 0 && i + 1 < argc) {
            seqlen_k = atoi(argv[++i]);
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
                      << "\nDENSE (non-varlen) format test\n"
                      << "\nNOTE: This build only supports:\n"
                      << "  - FP16 with head_dim <= 128 (default)\n"
                      << "  - FP8 E4M3 with head_dim <= 96\n"
                      << "  - BF16 is NOT supported\n"
                      << "\nOptions:\n"
                      << "  --batch_size N       Batch size (default: 2)\n"
                      << "  --seqlen_q N         Sequence length for Q (default: 512)\n"
                      << "  --seqlen_k N         Sequence length for K (default: 512)\n"
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

    head_size_v = head_size;

    std::cout << "FlashAttention DENSE Inference Test\n"
              << "====================================\n"
              << "Batch size: " << batch_size << "\n"
              << "Seqlen Q: " << seqlen_q << "\n"
              << "Seqlen K: " << seqlen_k << "\n"
              << "Num heads: " << num_heads << "\n"
              << "Num heads K: " << num_heads_k << "\n"
              << "Head size: " << head_size << "\n"
              << "Data type: " << (is_e4m3 ? "FP8" : (is_bf16 ? "BF16" : "FP16")) << "\n"
              << "Causal: " << (is_causal ? "Yes" : "No") << "\n"
              << "Format: DENSE (non-varlen)\n\n";

    // Total tokens
    int total_q = batch_size * seqlen_q;
    int total_k = batch_size * seqlen_k;

    std::cout << "Total Q tokens: " << total_q << "\n";
    std::cout << "Total K tokens: " << total_k << "\n\n";

    // Allocate device memory
    size_t dtype_size = is_e4m3 ? 1 : 2;  // FP8=1 byte, FP16/BF16=2 bytes

    // Q: (batch, seqlen_q, num_heads, head_size)
    void *d_q, *d_k, *d_v, *d_out;
    void *d_softmax_lse;
    void *d_scheduler_metadata;

    cudaMalloc(&d_q, total_q * num_heads * head_size * dtype_size);
    cudaMalloc(&d_k, total_k * num_heads_k * head_size * dtype_size);
    cudaMalloc(&d_v, total_k * num_heads_k * head_size_v * dtype_size);
    cudaMalloc(&d_out, total_q * num_heads * head_size_v * dtype_size);

    // softmax_lse: (batch, num_heads, seqlen_q) for dense
    cudaMalloc(&d_softmax_lse, batch_size * num_heads * seqlen_q * sizeof(float));

    // Scheduler metadata (still needed for some optimizations)
    cudaMalloc(&d_scheduler_metadata, 1024 * sizeof(int));
    cudaMemset(d_scheduler_metadata, 0, 1024 * sizeof(int));

    // Initialize random data
    std::cout << "Initializing random data...\n";
    init_random_data_gpu(d_q, total_q * num_heads * head_size, !is_bf16 && !is_e4m3, is_bf16, is_e4m3);
    init_random_data_gpu(d_k, total_k * num_heads_k * head_size, !is_bf16 && !is_e4m3, is_bf16, is_e4m3);
    init_random_data_gpu(d_v, total_k * num_heads_k * head_size_v, !is_bf16 && !is_e4m3, is_bf16, is_e4m3);

    // Calculate strides for DENSE format
    // Layout: (batch, seqlen, num_heads, head_size)
    int64_t q_batch_stride = seqlen_q * num_heads * head_size;
    int64_t q_row_stride = num_heads * head_size;
    int64_t q_head_stride = head_size;

    int64_t k_batch_stride = seqlen_k * num_heads_k * head_size;
    int64_t k_row_stride = num_heads_k * head_size;
    int64_t k_head_stride = head_size;

    int64_t v_batch_stride = seqlen_k * num_heads_k * head_size_v;
    int64_t v_row_stride = num_heads_k * head_size_v;
    int64_t v_head_stride = head_size_v;

    int64_t o_batch_stride = seqlen_q * num_heads * head_size_v;
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
        nullptr, nullptr, nullptr,  // cu_seqlens (nullptr = DENSE format)
        nullptr, nullptr,  // seqused_q, seqused_k
        seqlen_q, seqlen_k,  // max_seqlen (for dense, these are the actual seqlens)
        nullptr, nullptr, nullptr,  // page_table, kv_batch_idx, leftpad_k
        nullptr, nullptr, nullptr,  // rotary
        nullptr, nullptr, nullptr,  // FP8 descale (optional)
        d_scheduler_metadata,  // scheduler_metadata
        is_bf16, is_e4m3,
        batch_size, seqlen_q, num_heads, head_size,  // Q dims (seqlen_q is per-sequence length)
        batch_size, seqlen_k, num_heads_k, head_size_v,  // K dims
        0, 0, 0, 0,  // paged KV dims, rotary dim
        q_batch_stride, q_row_stride, q_head_stride,  // Q strides
        k_batch_stride, k_row_stride, k_head_stride,  // K strides
        v_batch_stride, v_row_stride, v_head_stride,  // V strides
        o_batch_stride, o_row_stride, o_head_stride,  // O strides
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
    // Attention FLOPs ≈ 4 * batch * seqlen_q * seqlen_k * head_size * num_heads
    double flops = 4.0 * batch_size * seqlen_q * seqlen_k * head_size * num_heads;
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
    cudaFree(d_softmax_lse);
    cudaFree(d_scheduler_metadata);
    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::cout << "\nTest completed successfully!\n";

    return 0;
}
