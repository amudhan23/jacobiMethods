#include <Eigen/Core>       // For Eigen::VectorXf, Eigen::Ref
#include <Eigen/SparseCore>


    Eigen::VectorXf solve_with_l2_jacobi(
    const Eigen::SparseMatrix<float, Eigen::RowMajor>& A,
    Eigen::SparseMatrix<float, Eigen::RowMajor> S2_host,
    const Eigen::VectorXf& b,
    int max_iter,
    float tol);