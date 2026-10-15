cmake_minimum_required(VERSION 3.18)
project(sparse_format_test LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find Eigen3
find_package(Eigen3 REQUIRED)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# CPU version
# add_executable(test_sparse_format test_sparse_format.cpp)
# target_link_libraries(test_sparse_format Eigen3::Eigen)

# GPU version
# 

add_executable(main main.cpp cu_sparse.cpp csr_matrix.cpp aggregated_jacobi.cu)
target_link_libraries(main
    Eigen3::Eigen
    CUDA::cudart
    CUDA::cusparse
)

# A-Jacobi solver with cuSPARSE SpMV
# add_executable(main main.cpp a-jacobi_diagonal.cu cu_sparse_spmv.cpp)
# target_link_libraries(a_jacobi_diagonal
#     Eigen3::Eigen
#     CUDA::cudart
#     CUDA::cusparse
# )

# message(STATUS "CPU test executable: test_jacobi_cpu")
message(STATUS "GPU test executable: jacobi_diagonal")
message(STATUS "A-Jacobi executable: a_jacobi_diagonal")
message(STATUS "Eigen3 include dir: ${EIGEN3_INCLUDE_DIR}")
message(STATUS "CUDA version: ${CUDAToolkit_VERSION}")
