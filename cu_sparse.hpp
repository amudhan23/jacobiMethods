#include <Eigen/Core>       // For Eigen::VectorXf, Eigen::Ref
#include <Eigen/SparseCore>

Eigen::SparseMatrix<float, Eigen::RowMajor> cuSparseMultiply(const Eigen::SparseMatrix<float, Eigen::RowMajor>& A);
Eigen::SparseMatrix<float, Eigen::RowMajor> square_csr_matrix_example(const Eigen::SparseMatrix<float, Eigen::RowMajor>& A);