# Flash Attention Standalone - Build Instructions

## 概述

本目录包含Flash Attention 3的独立版本（无PyTorch依赖），专为NVIDIA Hopper架构(SM90)优化。

## 文件说明

- `flash_api_standalone.cpp` - 独立API实现（去除PyTorch依赖）
- `test_varlen_inference.cpp` - 变长序列推理测试程序
- `CMakeLists.txt` - 完整版本CMake配置（包含所有kernel）
- `build_full.sh` - 完整编译脚本
- `build_minimal.sh` - 最小化编译脚本（用于快速测试）
- `compile_simple.sh` - 单kernel编译脚本（用于调试）

## 编译要求

### 硬件要求
- NVIDIA GPU: Hopper架构 (H100, H800等，计算能力9.0)
- 内存:
  - 完整编译: 至少32GB RAM
  - 最小化编译: 至少16GB RAM

### 软件要求
- CUDA Toolkit 12.x
- CMake >= 3.18
- GCC/G++ >= 11.0
- CUTLASS 3.x (已包含在 `../csrc/cutlass/include`)

## 编译选项

### 选项1: 完整编译（推荐用于生产环境）

包含所有kernel配置：
- Head dimensions: 64, 96, 128, 192, 256
- Data types: FP16, BF16, FP8-E4M3
- 总计: 15个kernel文件

```bash
cd /path/to/flash-attention/hopper
./build_full.sh
```

编译时间: 约1-2小时（取决于CPU核心数和内存）

### 选项2: 最小化编译（推荐用于快速测试）

仅包含常用kernel配置：
- Head dimensions: 64, 128
- Data types: FP16, BF16
- 总计: 4个kernel文件

```bash
cd /path/to/flash-attention/hopper
./build_minimal.sh
```

编译时间: 约20-40分钟

### 选项3: 手动CMake编译

```bash
mkdir -p build
cd build

cmake .. \
    -DCMAKE_CUDA_ARCHITECTURES=90 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc

# 根据可用内存调整并行度
# 32GB RAM: make -j4
# 64GB RAM: make -j8
# 128GB RAM: make -j16
make -j4
```

## 编译输出

成功编译后会生成：

1. **libflash_attn_standalone.so** - 共享库
   - 可被C/C++程序链接使用
   - 位置: `build_full/` 或 `build_minimal/`

2. **test_varlen_inference** - 测试程序
   - 用于验证功能正确性
   - 运行: `./test_varlen_inference`

## 使用示例

### C++ 程序集成

```cpp
#include <cuda_runtime.h>

// 声明mha_fwd_standalone函数（见flash_api_standalone.cpp）
extern "C" void mha_fwd_standalone(
    void* q_ptr, void* k_ptr, void* v_ptr,
    // ... 其他参数
    cudaStream_t stream
);

int main() {
    // 分配CUDA内存
    void *d_q, *d_k, *d_v, *d_out, *d_lse;
    cudaMalloc(&d_q, batch * seqlen * heads * headdim * sizeof(half));
    // ... 分配其他张量

    // 调用Flash Attention
    mha_fwd_standalone(
        d_q, d_k, d_v,
        nullptr, nullptr, nullptr,  // k_new, v_new, q_v
        d_out,
        nullptr, nullptr, nullptr,  // cu_seqlens
        nullptr, nullptr,           // seqused
        0, 0,                       // max_seqlen
        // ... 其他参数
        false, false,               // is_bf16, is_e4m3
        batch, seqlen, heads, headdim,
        // ... 更多参数
        0                           // stream
    );

    cudaDeviceSynchronize();
    return 0;
}
```

编译链接：
```bash
g++ -std=c++17 your_program.cpp \
    -I/usr/local/cuda/include \
    -L./build_full \
    -lflash_attn_standalone \
    -lcudart \
    -o your_program
```

## 故障排除

### 问题1: 编译时内存不足

**症状**: 编译过程中进程被killed

**解决方案**:
1. 使用 `build_minimal.sh` 而非 `build_full.sh`
2. 减少并行编译任务: `make -j1` 或 `make -j2`
3. 增加swap空间
4. 在内存更大的服务器上编译

### 问题2: CUDA架构不匹配

**症状**: `error: No kernel image is available for execution on the device`

**解决方案**:
确保GPU是Hopper架构（SM90）。可以检查：
```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```
应该显示 `9.0`

### 问题3: CUTLASS头文件找不到

**症状**: `fatal error: cutlass/xxx.h: No such file or directory`

**解决方案**:
确保CUTLASS在正确路径：
```bash
ls ../csrc/cutlass/include/cutlass/
```

### 问题4: 编译时间过长

**解决方案**:
1. 使用 `build_minimal.sh` 快速测试
2. 增加并行编译任务（如果内存足够）: `make -j8`
3. 只编译需要的kernel（修改CMakeLists.txt）

## 性能调优

### 编译优化级别

当前使用 `-O3` 优化。如需更快编译（但运行时性能略低）：

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug
# 或
cmake .. -DCMAKE_CUDA_FLAGS="-O2"
```

### Kernel配置选择

根据应用需求选择必要的kernel：

```cmake
# 仅编译hdim128 + FP16
set(KERNEL_SOURCES
    instantiations/flash_fwd_hdim128_fp16_sm90.cu
)
```

## 技术支持

- Flash Attention 3 官方仓库: https://github.com/Dao-AILab/flash-attention
- CUTLASS 官方文档: https://github.com/NVIDIA/cutlass

## License

遵循原始Flash Attention项目的BSD-3-Clause许可证。
