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

# 收集所有SM90 forward kernel文件
# 使用file(GLOB)自动收集所有forward kernel变体
# 这样可以包含所有Split/PagedKV/Softcap/PackGQA组合，避免undefined reference
file(GLOB KERNEL_SOURCES
    "instantiations/flash_fwd_*_sm90.cu"
)

# 过滤掉backward kernels（虽然文件名不会匹配，但以防万一）
list(FILTER KERNEL_SOURCES EXCLUDE REGEX "flash_bwd_")

# 打印找到的kernel文件数量
list(LENGTH KERNEL_SOURCES KERNEL_COUNT)
message(STATUS "Found ${KERNEL_COUNT} SM90 forward kernel files for compilation")

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
    CUDA::cudart
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
    CUDA::cudart
    CUDA::curand
)

# 安装规则
install(TARGETS flash_attn_standalone
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES flash_api_standalone.h
    DESTINATION include
)
