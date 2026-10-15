#include <Eigen/Core>       // For Eigen::VectorXf, Eigen::Ref
#include <Eigen/SparseCore>
#include "csr_matrix.hpp"

Eigen::SparseMatrix<float, Eigen::RowMajor> cuSparseMultiply(const Eigen::SparseMatrix<float, Eigen::RowMajor>& A);
Eigen::SparseMatrix<float, Eigen::RowMajor> square_csr_matrix_example(const Eigen::SparseMatrix<float, Eigen::RowMajor>& A);


void CuSparseMultiply(const CSRMatrix& A, const CSRMatrix& B, CSRMatrix& C);