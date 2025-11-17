#include "cu_sparse.hpp"
#include "jacobiMethods.cuh"
#include <iostream>

int main(){
    // Example usage of cuSparseMultiply
    Eigen::SparseMatrix<float, Eigen::RowMajor> A(4, 4);
    A.insert(0, 0) = 1.0f; A.insert(0, 2) = 2.0f; A.insert(0, 3) = 3.0f;
    A.insert(1, 1) = 4.0f;
    A.insert(2, 0) = 5.0f; A.insert(2, 2) = 6.0f; A.insert(2, 3) = 7.0f;
    A.insert(3, 1) = 8.0f; A.insert(3, 3) = 9.0f;
    A.makeCompressed();

    std::cout << "Input matrix A:" << std::endl;
    std::cout << Eigen::MatrixXf(A) << std::endl << std::endl;

    // Perform sparse matrix multiplication (A * A)
    Eigen::SparseMatrix<float, Eigen::RowMajor> C = cuSparseMultiply(A);

    // Print the resulting C matrix
    std::cout << "Result matrix C (A * A):" << std::endl;
    std::cout << Eigen::MatrixXf(C) << std::endl << std::endl;

    std::cout << "C matrix info:" << std::endl;
    std::cout << "Rows: " << C.rows() << ", Cols: " << C.cols()
              << ", Non-zeros: " << C.nonZeros() << std::endl;


    // Example usage of L2 Jacobi solver
    Eigen::VectorXf b(4);
    b << 1.0f, 2.0f, 3.0f, 4.0f;
    Eigen::VectorXf x(4);
    x.setZero();

    // For demonstration, use C as S2_host
    Eigen::SparseMatrix<float, Eigen::RowMajor> S2_host = C;

    int max_iter = 100;
    float tol = 1e-5;

    Eigen::VectorXf x_sol = solve_with_l2_jacobi(A, S2_host, b, max_iter, tol);

    std::cout << "Solution vector x:" << std::endl;
    std::cout << x_sol << std::endl;

    return 0;
}