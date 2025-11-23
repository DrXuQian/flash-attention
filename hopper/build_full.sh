#!/bin/bash

# 完整编译脚本 - 包含所有kernel配置
# 注意：这需要大量内存和时间（可能需要1-2小时）
# 推荐在有至少32GB内存的服务器上运行

set -e

echo "================================================"
echo "Flash Attention Standalone - Full Build Script"
echo "================================================"
echo ""
echo "This will compile all kernel variants:"
echo "- Head dimensions: 64, 96, 128, 192, 256"
echo "- Data types: FP16, BF16, FP8-E4M3"
echo "- Total: 15 kernel files"
echo ""
echo "Requirements:"
echo "- CUDA Toolkit 12.x"
echo "- At least 32GB RAM recommended"
echo "- Estimated time: 1-2 hours"
echo ""

# 检查CUDA是否可用
if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc not found. Please ensure CUDA is installed."
    exit 1
fi

echo "CUDA version:"
nvcc --version
echo ""

# 创建build目录
BUILD_DIR="build_full"
echo "Creating build directory: ${BUILD_DIR}"
mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

# 清理之前的构建
echo "Cleaning previous build..."
rm -rf *

# 运行CMake配置
echo ""
echo "Running CMake configuration..."
cmake .. \
    -DCMAKE_CUDA_ARCHITECTURES=90 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc

if [ $? -ne 0 ]; then
    echo "Error: CMake configuration failed"
    exit 1
fi

# 编译
echo ""
echo "Starting compilation..."
echo "Using parallel jobs: $(nproc) cores"
echo ""

# 使用较少的并行任务以避免内存不足
# 可以根据服务器内存调整 -j 参数
# 32GB RAM: -j4
# 64GB RAM: -j8
# 128GB RAM: -j16

PARALLEL_JOBS=4
echo "Compiling with ${PARALLEL_JOBS} parallel jobs..."
make -j${PARALLEL_JOBS}

if [ $? -ne 0 ]; then
    echo ""
    echo "Error: Compilation failed"
    echo "Try reducing parallel jobs or compiling with: make -j1"
    exit 1
fi

echo ""
echo "================================================"
echo "Compilation successful!"
echo "================================================"
echo ""
echo "Output files:"
echo "  Shared library: ${BUILD_DIR}/libflash_attn_standalone.so"
echo "  Test program:   ${BUILD_DIR}/test_varlen_inference"
echo ""
echo "To run the test:"
echo "  cd ${BUILD_DIR} && ./test_varlen_inference"
echo ""
