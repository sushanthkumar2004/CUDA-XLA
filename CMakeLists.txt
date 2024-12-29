cmake_minimum_required(VERSION 3.28)
project(xla LANGUAGES CXX)

# CUDA configuration
option(USE_CUDA "Use CUDA" ON)
if(USE_CUDA)
    enable_language(CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
endif()

# Source file collection
file(GLOB_RECURSE cpu_src 
    "${CMAKE_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/*.cxx"
    "${CMAKE_SOURCE_DIR}/src/*.cc"
)

if(USE_CUDA)
    file(GLOB_RECURSE gpu_src 
        "${CMAKE_SOURCE_DIR}/src/*.cu"
        "${CMAKE_SOURCE_DIR}/src/*.cuh"
    )
endif()

# Debug output
message(STATUS "CPU Source Files: ${cpu_src}")
if(USE_CUDA)
    message(STATUS "GPU Source Files: ${gpu_src}")
endif()

# Main executable
add_executable(xla ${cpu_src} ${gpu_src})
target_include_directories(xla 
    PRIVATE 
        "${CMAKE_SOURCE_DIR}/include"
)

# CUDA-specific configurations
if(USE_CUDA)
    set_target_properties(xla PROPERTIES 
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_compile_definitions(xla PRIVATE USE_CUDA)
endif()

# Set C++ standard
set_target_properties(xla PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)