#!/bin/bash

# 简化的编译脚本 - 只编译一个最小的kernel配置

set -e

CUDA_PATH=/usr/local/cuda
CUTLASS_PATH=../csrc/cutlass/include
FA_PATH=..

echo "Step 1: Compiling flash_api_standalone.cpp..."
g++ -c -std=c++17 -O3 \
  -I${CUDA_PATH}/include \
  -I${FA_PATH} \
  -I${CUTLASS_PATH} \
  -DFLASHATTENTION_DISABLE_BACKWARD \
  flash_api_standalone.cpp \
  -o flash_api_standalone.o

echo "Step 2: Compiling minimal kernel (hdim64_fp16_sm90)..."
nvcc -std=c++17 -O2 \
  -arch=sm_90 \
  --expt-relaxed-constexpr \
  -I${CUDA_PATH}/include \
  -I${FA_PATH} \
  -I${CUTLASS_PATH} \
  -I. \
  -DFLASHATTENTION_DISABLE_BACKWARD \
  -c instantiations/flash_fwd_hdim64_fp16_sm90.cu \
  -o flash_fwd_hdim64_fp16_sm90.o \
  --ptxas-options=-v \
  2>&1 | tee kernel_compile.log

echo "Step 3: Creating shared library..."
nvcc -shared -O3 \
  -arch=sm_90 \
  flash_api_standalone.o \
  flash_fwd_hdim64_fp16_sm90.o \
  -lcudart \
  -o libflash_attn_standalone.so

echo ""
echo "Compilation successful!"
echo "Library: libflash_attn_standalone.so"
