cmake_minimum_required(VERSION 3.18)
project(FlashAttentionStandalone CUDA CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# 启用CUDA
enable_language(CUDA)

# 查找CUDA包
find_package(CUDAToolkit REQUIRED)

# CUDA架构设置 (Hopper SM90)
set(CMAKE_CUDA_ARCHITECTURES 90)

# 路径设置
set(CUTLASS_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../csrc/cutlass/include")
set(FA_PATH "${CMAKE_CURRENT_SOURCE_DIR}/..")

# Include目录
include_directories(
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${FA_PATH}
    ${CUTLASS_PATH}
)

# 库目录
link_directories(
    ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
    /usr/local/cuda/lib64
)

# 编译选项
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --expt-relaxed-constexpr")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# 定义宏
add_definitions(-DFLASHATTENTION_DISABLE_BACKWARD)
# 只编译SM90代码，禁用SM80/SM86分支，避免链接SM80/SM86 kernel
add_definitions(-DFLASHATTENTION_DISABLE_SM8x)

# 方案2：只包含常用的kernel组合
# 编译时间: 10-20分钟，内存需求: 16GB
# 包含最常用的功能组合，避免编译所有280个kernel文件

set(KERNEL_SOURCES
    # 基础kernels（无特殊功能）- 必需
    instantiations/flash_fwd_hdim64_fp16_sm90.cu
    instantiations/flash_fwd_hdim64_bf16_sm90.cu
    instantiations/flash_fwd_hdim64_e4m3_sm90.cu
    instantiations/flash_fwd_hdim96_fp16_sm90.cu
    instantiations/flash_fwd_hdim96_bf16_sm90.cu
    instantiations/flash_fwd_hdim96_e4m3_sm90.cu
    instantiations/flash_fwd_hdim128_fp16_sm90.cu
    instantiations/flash_fwd_hdim128_bf16_sm90.cu
    instantiations/flash_fwd_hdim128_e4m3_sm90.cu
    instantiations/flash_fwd_hdim192_bf16_sm90.cu
    instantiations/flash_fwd_hdim192_128_bf16_sm90.cu
    instantiations/flash_fwd_hdim256_bf16_sm90.cu
    instantiations/flash_fwd_hdim256_e4m3_sm90.cu

    # PackGQA variants（用于GQA/MQA）- 常用
    instantiations/flash_fwd_hdim64_fp16_packgqa_sm90.cu
    instantiations/flash_fwd_hdim64_bf16_packgqa_sm90.cu
    instantiations/flash_fwd_hdim128_fp16_packgqa_sm90.cu
    instantiations/flash_fwd_hdim128_bf16_packgqa_sm90.cu
    instantiations/flash_fwd_hdim128_e4m3_packgqa_sm90.cu
    instantiations/flash_fwd_hdim192_e4m3_packgqa_sm90.cu

    # Softcap variants（用于logit capping）- 较常用
    instantiations/flash_fwd_hdim64_e4m3_softcap_sm90.cu
    instantiations/flash_fwd_hdim96_bf16_softcap_sm90.cu
    instantiations/flash_fwd_hdim128_e4m3_softcap_sm90.cu
    instantiations/flash_fwd_hdim256_e4m3_softcap_sm90.cu

    # Softcap + PackGQA组合 - 常用
    instantiations/flash_fwd_hdim96_bf16_softcap_packgqa_sm90.cu
    instantiations/flash_fwd_hdim96_fp16_softcap_packgqa_sm90.cu
    instantiations/flash_fwd_hdim128_fp16_softcap_packgqa_sm90.cu
    instantiations/flash_fwd_hdim128_bf16_softcap_packgqa_sm90.cu
    instantiations/flash_fwd_hdim192_128_bf16_softcap_packgqa_sm90.cu
    instantiations/flash_fwd_hdim192_e4m3_softcap_packgqa_sm90.cu

    # Split variants（用于长序列）- 常用
    instantiations/flash_fwd_hdim64_fp16_split_sm90.cu
    instantiations/flash_fwd_hdim96_fp16_split_sm90.cu
    instantiations/flash_fwd_hdim96_bf16_split_sm90.cu
    instantiations/flash_fwd_hdim128_e4m3_split_sm90.cu
    instantiations/flash_fwd_hdim192_bf16_split_sm90.cu

    # Split + Softcap组合 - 常用于长序列with logit capping
    instantiations/flash_fwd_hdim96_e4m3_split_softcap_sm90.cu
    instantiations/flash_fwd_hdim128_fp16_split_softcap_sm90.cu
    instantiations/flash_fwd_hdim128_e4m3_split_softcap_sm90.cu
    instantiations/flash_fwd_hdim192_128_e4m3_split_softcap_sm90.cu
    instantiations/flash_fwd_hdim256_fp16_split_softcap_sm90.cu
    instantiations/flash_fwd_hdim256_bf16_split_softcap_sm90.cu

    # Paged KV variants（用于KV cache管理）- 常用
    instantiations/flash_fwd_hdim128_bf16_paged_sm90.cu
    instantiations/flash_fwd_hdim128_e4m3_paged_sm90.cu
    instantiations/flash_fwd_hdim192_bf16_paged_sm90.cu

    # Paged + Softcap组合
    instantiations/flash_fwd_hdim96_bf16_paged_softcap_sm90.cu
    instantiations/flash_fwd_hdim128_fp16_paged_softcap_sm90.cu
    instantiations/flash_fwd_hdim128_e4m3_paged_softcap_sm90.cu
    instantiations/flash_fwd_hdim192_128_e4m3_paged_softcap_sm90.cu
    instantiations/flash_fwd_hdim256_bf16_paged_softcap_sm90.cu
    instantiations/flash_fwd_hdim256_e4m3_paged_softcap_sm90.cu

    # Paged + Split组合（用于长序列 + KV cache）
    instantiations/flash_fwd_hdim96_bf16_paged_split_sm90.cu
    instantiations/flash_fwd_hdim128_bf16_paged_split_sm90.cu
    instantiations/flash_fwd_hdim128_fp16_paged_split_sm90.cu
    instantiations/flash_fwd_hdim192_e4m3_paged_split_sm90.cu

    # Paged + Split + Softcap组合（全功能）
    instantiations/flash_fwd_hdim64_fp16_paged_split_softcap_sm90.cu
    instantiations/flash_fwd_hdim96_fp16_paged_split_softcap_sm90.cu
    instantiations/flash_fwd_hdim128_fp16_paged_split_softcap_sm90.cu
    instantiations/flash_fwd_hdim128_e4m3_paged_split_softcap_sm90.cu
    instantiations/flash_fwd_hdim192_e4m3_paged_split_softcap_sm90.cu

    # 特殊hdimV变体（用于不同的head dimension）
    instantiations/flash_fwd_hdim64_256_fp16_sm90.cu
    instantiations/flash_fwd_hdim64_256_bf16_sm90.cu
    instantiations/flash_fwd_hdim64_256_fp16_softcap_sm90.cu
    instantiations/flash_fwd_hdim64_256_fp16_split_softcap_sm90.cu
    instantiations/flash_fwd_hdim64_512_fp16_split_softcap_sm90.cu
    instantiations/flash_fwd_hdim64_512_bf16_split_softcap_sm90.cu
    instantiations/flash_fwd_hdim192_128_fp16_softcap_sm90.cu
)

list(LENGTH KERNEL_SOURCES KERNEL_COUNT)
message(STATUS "Compiling ${KERNEL_COUNT} commonly-used SM90 forward kernel variants")

# 直接创建可执行文件，不使用动态库
# 这样可以让链接器在编译时解决所有符号引用，避免undefined reference
add_executable(test_varlen_inference
    test_varlen_inference.cpp
    flash_api_standalone.cpp
    ${KERNEL_SOURCES}
)

# 设置CUDA分离编译
set_target_properties(test_varlen_inference PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(test_varlen_inference
    CUDA::cudart
    CUDA::curand
)

# 安装规则
install(TARGETS test_varlen_inference
    RUNTIME DESTINATION bin
)

install(FILES flash_api_standalone.h
    DESTINATION include
)
