#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdexcept>

#include "cu_sparse_spmv.hpp"

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        throw std::runtime_error("CUDA error");                                \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        throw std::runtime_error("CUSPARSE error");                            \
    }                                                                          \
}

// Sparse Matrix-Vector Multiplication: y = alpha * A * x + beta * y
void cuSparseSpMV(const Eigen::SparseMatrix<float, Eigen::RowMajor>& A,
                  const Eigen::VectorXf& x,
                  Eigen::VectorXf& y,
                  float alpha,
                  float beta) {

    const int num_rows = A.rows();
    const int num_cols = A.cols();
    const int nnz = A.nonZeros();

    if (x.size() != num_cols) {
        throw std::runtime_error("SpMV: x vector size mismatch");
    }
    if (y.size() != num_rows) {
        y.resize(num_rows);
    }

    // Device memory pointers
    int *dA_csrOffsets = nullptr, *dA_columns = nullptr;
    float *dA_values = nullptr;
    float *dx = nullptr, *dy = nullptr;

    // Allocate device memory
    CHECK_CUDA(cudaMalloc((void**)&dA_csrOffsets, (num_rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&dA_columns, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&dA_values, nnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&dx, num_cols * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&dy, num_rows * sizeof(float)));

    // Copy matrix A to device
    CHECK_CUDA(cudaMemcpy(dA_csrOffsets, A.outerIndexPtr(),
                          (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_columns, A.innerIndexPtr(),
                          nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_values, A.valuePtr(),
                          nnz * sizeof(float), cudaMemcpyHostToDevice));

    // Copy vector x to device
    CHECK_CUDA(cudaMemcpy(dx, x.data(), num_cols * sizeof(float), cudaMemcpyHostToDevice));

    // Copy vector y to device (needed if beta != 0)
    if (beta != 0.0f) {
        CHECK_CUDA(cudaMemcpy(dy, y.data(), num_rows * sizeof(float), cudaMemcpyHostToDevice));
    }

    // Create cuSPARSE handle
    cusparseHandle_t handle = nullptr;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    // Create sparse matrix descriptor
    cusparseSpMatDescr_t matA;
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, num_rows, num_cols, nnz,
                                     dA_csrOffsets, dA_columns, dA_values,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    // Create dense vector descriptors
    cusparseDnVecDescr_t vecX, vecY;
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, num_cols, dx, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, num_rows, dy, CUDA_R_32F));

    // Allocate external buffer for SpMV
    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));

    void* dBuffer = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&dBuffer, bufferSize));

    // Execute SpMV: y = alpha * A * x + beta * y
    CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                                CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(y.data(), dy, num_rows * sizeof(float), cudaMemcpyDeviceToHost));

    // Cleanup
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));
    CHECK_CUSPARSE(cusparseDestroy(handle));

    CHECK_CUDA(cudaFree(dBuffer));
    CHECK_CUDA(cudaFree(dA_csrOffsets));
    CHECK_CUDA(cudaFree(dA_columns));
    CHECK_CUDA(cudaFree(dA_values));
    CHECK_CUDA(cudaFree(dx));
    CHECK_CUDA(cudaFree(dy));
}

// GPU-only SpMV: operates entirely on device pointers
// Assumes handle, descriptors, and buffer are already set up
void cuSparseSpMV_device(cusparseHandle_t handle,
                         cusparseSpMatDescr_t matA,
                         cusparseDnVecDescr_t vecX,
                         cusparseDnVecDescr_t vecY,
                         float alpha,
                         float beta,
                         void* dBuffer) {
    // Execute SpMV: y = alpha * A * x + beta * y
    CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                                CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
}
