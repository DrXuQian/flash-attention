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
# 启用CUTLASS SM90a支持 (WGMMA指令)
add_definitions(-DCUTE_ARCH_MMA_SM90A_ENABLED)
# 禁用CUTLASS调试日志避免__host__ __device__冲突
add_definitions(-DCUTLASS_DEBUG_TRACE_LEVEL=0)

# 极简版本：只支持两种配置
# - FP8 E4M3, head_dim=96
# - FP16, head_dim=128
# 编译时间: 1-2分钟，内存需求: 8GB

# 自动收集这两种配置的所有变体（包括Split/PackGQA/Softcap/Paged等组合）
file(GLOB KERNEL_FP16_128
    "instantiations/flash_fwd_hdim128_fp16*_sm90.cu"
)
file(GLOB KERNEL_FP8_96
    "instantiations/flash_fwd_hdim96_e4m3*_sm90.cu"
)

set(KERNEL_SOURCES
    ${KERNEL_FP16_128}
    ${KERNEL_FP8_96}
)

list(LENGTH KERNEL_SOURCES KERNEL_COUNT)
message(STATUS "Compiling ${KERNEL_COUNT} kernel variants (FP16 hdim128 + FP8 hdim96)")

# 直接创建可执行文件，不使用动态库
# 这样可以让链接器在编译时解决所有符号引用，避免undefined reference

# Varlen (variable-length) test
add_executable(test_varlen_inference
    test_varlen_inference.cpp
    flash_api_standalone.cpp
    flash_prepare_scheduler.cu
    ${KERNEL_SOURCES}
)

set_target_properties(test_varlen_inference PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(test_varlen_inference
    CUDA::cudart
    CUDA::curand
)

# Dense (fixed-length) test
add_executable(test_dense_inference
    test_dense_inference.cpp
    flash_api_standalone.cpp
    flash_prepare_scheduler.cu
    ${KERNEL_SOURCES}
)

set_target_properties(test_dense_inference PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(test_dense_inference
    CUDA::cudart
    CUDA::curand
)

# 安装规则
install(TARGETS test_varlen_inference test_dense_inference
    RUNTIME DESTINATION bin
)

install(FILES flash_api_standalone.h
    DESTINATION include
)
