#include <Eigen/Core>       // For Eigen::VectorXf, Eigen::Ref
#include <Eigen/SparseCore>
#include "csr_matrix.hpp"

    Eigen::VectorXf solve_with_l2_jacobi(
    const Eigen::SparseMatrix<float, Eigen::RowMajor>& A,
    Eigen::SparseMatrix<float, Eigen::RowMajor> S2_host,
    const Eigen::VectorXf& b,
    int max_iter,
    float tol);


    bool a_jacobi(int n, int max_iter, float abs_tol, float rel_tol,
              const CSRMatrix& B, const float* D,
              const float* b, float* x);




