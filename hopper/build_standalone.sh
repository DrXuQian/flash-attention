#!/bin/bash

# 创建build目录
mkdir -p build_standalone
cd build_standalone

# 运行CMake配置
cmake .. \
    -DCMAKE_CUDA_ARCHITECTURES=90 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc

# 编译 (使用4个并行任务以避免内存不足)
make -j4

echo ""
echo "编译完成！"
echo "动态库: build_standalone/libflash_attn_standalone.so"
echo "测试程序: build_standalone/test_varlen_inference"
