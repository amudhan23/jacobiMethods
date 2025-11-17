#ifndef CU_SPARSE_SPMV_HPP
#define CU_SPARSE_SPMV_HPP

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <cusparse.h>

// Sparse Matrix-Vector Multiplication: y = alpha * A * x + beta * y
// Uses cuSPARSE for GPU acceleration (host-based wrapper)
void cuSparseSpMV(const Eigen::SparseMatrix<float, Eigen::RowMajor>& A,
                  const Eigen::VectorXf& x,
                  Eigen::VectorXf& y,
                  float alpha = 1.0f,
                  float beta = 0.0f);

// GPU-only SpMV: operates entirely on device pointers
// y = alpha * A * x + beta * y
// All pointers must be device pointers!
void cuSparseSpMV_device(cusparseHandle_t handle,
                         cusparseSpMatDescr_t matA,
                         cusparseDnVecDescr_t vecX,
                         cusparseDnVecDescr_t vecY,
                         float alpha,
                         float beta,
                         void* dBuffer);

#endif // CU_SPARSE_SPMV_HPP
