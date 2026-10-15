#include "cu_sparse.hpp"
#include "jacobiMethods.cuh"
#include <iostream>
#include "csr_matrix.hpp"
#include "sparse_matrix_interop.hpp"


#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        throw std::runtime_error("CUDA error");                                \
    }                                                                          \
}

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

     Eigen::SparseMatrix<float, Eigen::RowMajor> B(4, 4);

    // Insert non-zero elements
    B.insert(0, 1) = 10.0f;
    B.insert(0, 3) = 11.0f;
    
    B.insert(1, 0) = 12.0f;
    B.insert(1, 1) = 13.0f;
    B.insert(1, 2) = 14.0f;
    
    B.insert(2, 2) = 15.0f;
    
    B.insert(3, 0) = 16.0f;
    B.insert(3, 3) = 17.0f;

    // Call makeCompressed() after all insertions
    B.makeCompressed();   

    CSRMatrix A_csr = make_csr_from_eigen(A);
    CSRMatrix B_csr = make_csr_from_eigen(B);

    CSRMatrix C_csr;

    CSRMatrix *d_A_csr, *d_B_csr, *d_C_csr;
    CHECK_CUDA( cudaMalloc((void**)&d_A_csr, sizeof(CSRMatrix)) )
    CHECK_CUDA( cudaMalloc((void**)&d_B_csr, sizeof(CSRMatrix)) )
    CHECK_CUDA( cudaMalloc((void**)&d_C_csr, sizeof(CSRMatrix)) )   
    
    CuSparseMultiply(d_A_csr, d_B_csr, d_C_csr);

    // Copy result back to host
    CHECK_CUDA( cudaMemcpy(&C_csr, d_C_csr, sizeof(CSRMatrix), cudaMemcpyDeviceToHost) );

    // Free device memory
    CHECK_CUDA( cudaFree(d_A_csr) );
    CHECK_CUDA( cudaFree(d_B_csr) );
    CHECK_CUDA( cudaFree(d_C_csr) );

    std::cout << "Result matrix C (A * B):" << std::endl;
    for(int i = 0; i < C_csr.row_num; i++) {
        for(int j = C_csr.d_row_ptr[i]; j < C_csr.d_row_ptr[i + 1]; j++) {
            std::cout << "C(" << i << ", " << C_csr.d_col_idx[j] << ") = " << C_csr.d_values[j] << std::endl;
        }
    }

    return 0;
}


// Perform sparse matrix multiplication (A * A)
    // Eigen::SparseMatrix<float, Eigen::RowMajor> C = cuSparseMultiply(A);

    // // Print the resulting C matrix
    // std::cout << "Result matrix C (A * A):" << std::endl;
    // std::cout << Eigen::MatrixXf(C) << std::endl << std::endl;

    // std::cout << "C matrix info:" << std::endl;
    // std::cout << "Rows: " << C.rows() << ", Cols: " << C.cols()
    //           << ", Non-zeros: " << C.nonZeros() << std::endl;


    // // Example usage of L2 Jacobi solver
    // Eigen::VectorXf b(4);
    // b << 1.0f, 2.0f, 3.0f, 4.0f;
    // Eigen::VectorXf x(4);
    // x.setZero();

    // // For demonstration, use C as S2_host
    // Eigen::SparseMatrix<float, Eigen::RowMajor> S2_host = C;

    // int max_iter = 100;
    // float tol = 1e-5;

    // Eigen::VectorXf x_sol = solve_with_l2_jacobi(A, S2_host, b, max_iter, tol);

    // std::cout << "Solution vector x:" << std::endl;
    // std::cout << x_sol << std::endl;