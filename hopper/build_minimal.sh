#!/bin/bash

# 最小化编译脚本 - 只编译必要的kernel配置
# 适用于内存受限的服务器或快速测试

set -e

echo "=================================================="
echo "Flash Attention Standalone - Minimal Build Script"
echo "=================================================="
echo ""
echo "This will compile only essential kernels:"
echo "- Head dimensions: 64, 128"
echo "- Data types: FP16, BF16"
echo "- Total: 4 kernel files"
echo ""
echo "Requirements:"
echo "- CUDA Toolkit 12.x"
echo "- At least 16GB RAM recommended"
echo "- Estimated time: 20-40 minutes"
echo ""

# 检查CUDA是否可用
if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc not found. Please ensure CUDA is installed."
    exit 1
fi

echo "CUDA version:"
nvcc --version
echo ""

# 创建临时CMakeLists.txt用于最小化编译
BUILD_DIR="build_minimal"
echo "Creating build directory: ${BUILD_DIR}"
mkdir -p ${BUILD_DIR}

# 创建最小化版本的CMakeLists.txt
cat > ${BUILD_DIR}/CMakeLists.txt << 'EOF'
cmake_minimum_required(VERSION 3.18)
project(FlashAttentionStandalone CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES 90)

set(CUTLASS_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../csrc/cutlass/include")
set(FA_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../..")

include_directories(
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${FA_PATH}
    ${CUTLASS_PATH}
)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
add_definitions(-DFLASHATTENTION_DISABLE_BACKWARD)

# 最小kernel集合
set(KERNEL_SOURCES
    ../instantiations/flash_fwd_hdim64_fp16_sm90.cu
    ../instantiations/flash_fwd_hdim64_bf16_sm90.cu
    ../instantiations/flash_fwd_hdim128_fp16_sm90.cu
    ../instantiations/flash_fwd_hdim128_bf16_sm90.cu
)

add_library(flash_kernels OBJECT ${KERNEL_SOURCES})
set_target_properties(flash_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

add_library(flash_attn_standalone SHARED
    ../flash_api_standalone.cpp
    $<TARGET_OBJECTS:flash_kernels>
)

target_link_libraries(flash_attn_standalone cudart)

add_executable(test_varlen_inference ../test_varlen_inference.cpp)
target_link_libraries(test_varlen_inference flash_attn_standalone cudart curand)
EOF

cd ${BUILD_DIR}

# 清理之前的构建
echo "Cleaning previous build..."
rm -rf CMakeCache.txt CMakeFiles Makefile cmake_install.cmake *.o *.so test_varlen_inference

# 运行CMake配置
echo ""
echo "Running CMake configuration..."
cmake . \
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
echo ""

# 单线程编译以节省内存
echo "Compiling with single thread to save memory..."
make -j1

if [ $? -ne 0 ]; then
    echo ""
    echo "Error: Compilation failed"
    exit 1
fi

echo ""
echo "=================================================="
echo "Minimal compilation successful!"
echo "=================================================="
echo ""
echo "Output files:"
echo "  Shared library: ${BUILD_DIR}/libflash_attn_standalone.so"
echo "  Test program:   ${BUILD_DIR}/test_varlen_inference"
echo ""
echo "Note: This minimal build only supports:"
echo "  - Head dimensions: 64, 128"
echo "  - Data types: FP16, BF16"
echo ""
echo "For full feature support, use build_full.sh"
echo ""
