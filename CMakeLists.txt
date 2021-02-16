cmake_minimum_required(VERSION 3.4...3.18)
project (pybind11_test)

find_package(CUDA REQUIRED)
add_subdirectory(pybind11)

include_directories(
    ${PYTHON_INCLUDE_DIRS}
    "pybind11/include"
    "dlpack/include/dlpack"
)

link_directories(
    /usr/local/cuda/lib64
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

cuda_add_library(pybind11_test SHARED
    src/main.cu
)

target_link_libraries(pybind11_test
    ${PYTHON_LIBRARIES}
    cudart
)

set_target_properties(pybind11_test PROPERTIES PREFIX "")

target_compile_definitions(pybind11_test PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

