cmake_minimum_required(VERSION 3.18)
project(FlashAttentionStandalone CUDA CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# 启用CUDA
enable_language(CUDA)

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

# 编译选项
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --expt-relaxed-constexpr")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# 定义宏
add_definitions(-DFLASHATTENTION_DISABLE_BACKWARD)

# 收集所有kernel实现文件 (完整版本，包含所有hdim和数据类型)
set(KERNEL_SOURCES
    # hdim64 kernels
    instantiations/flash_fwd_hdim64_fp16_sm90.cu
    instantiations/flash_fwd_hdim64_bf16_sm90.cu
    instantiations/flash_fwd_hdim64_e4m3_sm90.cu

    # hdim96 kernels
    instantiations/flash_fwd_hdim96_fp16_sm90.cu
    instantiations/flash_fwd_hdim96_bf16_sm90.cu
    instantiations/flash_fwd_hdim96_e4m3_sm90.cu

    # hdim128 kernels
    instantiations/flash_fwd_hdim128_fp16_sm90.cu
    instantiations/flash_fwd_hdim128_bf16_sm90.cu
    instantiations/flash_fwd_hdim128_e4m3_sm90.cu

    # hdim192 kernels
    instantiations/flash_fwd_hdim192_fp16_sm90.cu
    instantiations/flash_fwd_hdim192_bf16_sm90.cu
    instantiations/flash_fwd_hdim192_e4m3_sm90.cu

    # hdim256 kernels
    instantiations/flash_fwd_hdim256_fp16_sm90.cu
    instantiations/flash_fwd_hdim256_bf16_sm90.cu
    instantiations/flash_fwd_hdim256_e4m3_sm90.cu
)

# 创建kernel对象库
add_library(flash_kernels OBJECT ${KERNEL_SOURCES})
set_target_properties(flash_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# 创建standalone API动态库
add_library(flash_attn_standalone SHARED
    flash_api_standalone.cpp
    $<TARGET_OBJECTS:flash_kernels>
)

target_link_libraries(flash_attn_standalone
    cudart
)

# 设置输出目录
set_target_properties(flash_attn_standalone PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# 创建测试程序
add_executable(test_varlen_inference
    test_varlen_inference.cpp
)

target_link_libraries(test_varlen_inference
    flash_attn_standalone
    cudart
    curand
)

# 安装规则
install(TARGETS flash_attn_standalone
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES flash_api_standalone.cpp
    DESTINATION include
)
