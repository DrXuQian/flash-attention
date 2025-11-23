#!/bin/bash

# 设置路径
CUDA_PATH=/usr/local/cuda
CUTLASS_PATH=../csrc/cutlass/include
FA_PATH=..

# 编译 flash_api_standalone.cpp
echo "Compiling flash_api_standalone.cpp..."
nvcc -std=c++17 -O3 \
  -arch=sm_90 \
  --expt-relaxed-constexpr \
  -I${CUDA_PATH}/include \
  -I${FA_PATH} \
  -I${CUTLASS_PATH} \
  -DFLASHATTENTION_DISABLE_BACKWARD \
  -c flash_api_standalone.cpp \
  -o flash_api_standalone.o

# 编译 test_varlen_inference.cpp
echo "Compiling test_varlen_inference.cpp..."
nvcc -std=c++17 -O3 \
  -arch=sm_90 \
  --expt-relaxed-constexpr \
  -I${CUDA_PATH}/include \
  -I${FA_PATH} \
  -I${CUTLASS_PATH} \
  -c test_varlen_inference.cpp \
  -o test_varlen_inference.o

# 链接
echo "Linking..."
nvcc -std=c++17 -O3 \
  -arch=sm_90 \
  flash_api_standalone.o \
  test_varlen_inference.o \
  -L${CUDA_PATH}/lib64 \
  -lcudart -lcurand \
  -o test_varlen_inference

echo "Done! Executable: test_varlen_inference"
