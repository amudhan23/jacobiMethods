#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <iomanip>
#include "cu_sparse_spmv.hpp"

// CUDA error checking
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// cuSPARSE error checking
#define CUSPARSE_CHECK(call) \
do { \
    cusparseStatus_t err = call; \
    if (err != CUSPARSE_STATUS_SUCCESS) { \
        std::cerr << "cuSPARSE Error: " << cusparseGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// -----------------------------------------------------------------
// A-Jacobi Kernel: Damped diagonal-preconditioned Richardson
// x_new[i] = x_old[i] - omega * (residual[i] / diag[i])
// where residual = A*x_old - b
// -----------------------------------------------------------------
__global__ void a_jacobi_update_kernel(
    const float* diag,        // Diagonal elements of A
    const float* residual,    // residual = A*x - b
    const float* x_old,       // Current solution
    float* x_new,             // Updated solution
    float omega,              // Damping parameter
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        if (diag[i] != 0.0f) {
            // A-Jacobi update: x_new = x_old - omega * D^(-1) * residual
            x_new[i] = x_old[i] - omega * (residual[i] / diag[i]);
        } else {
            x_new[i] = x_old[i]; // Keep unchanged if diagonal is zero
        }
    }
}

// -----------------------------------------------------------------
// Kernel: Compute residual = A*x - b
// (A*x is computed on host using cuSPARSE SpMV, then uploaded)
// -----------------------------------------------------------------
__global__ void compute_residual_kernel(
    const float* Ax,
    const float* b,
    float* residual,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        residual[i] = Ax[i] - b[i];
    }
}

// -----------------------------------------------------------------
// GPU A-Jacobi Solver - OPTIMIZED (Pure GPU Implementation)
// Uses cuSPARSE SpMV entirely on GPU - NO memcpy inside loop!
// -----------------------------------------------------------------
void gpu_a_jacobi_solve(const Eigen::SparseMatrix<float, Eigen::RowMajor>& A,
                        const Eigen::VectorXf& b,
                        Eigen::VectorXf& x,
                        int max_iter,
                        float tol,
                        float omega = 0.67f) // Damping parameter
{
    int n = A.rows();
    int nnz = A.nonZeros();
    x.resize(n);
    x.setZero();

    // Extract diagonal of A
    Eigen::VectorXf D_host(n);
    for (int i = 0; i < n; ++i) {
        D_host(i) = A.coeff(i, i);
        if (D_host(i) == 0.0f) {
            std::cerr << "Warning: Zero diagonal at row " << i << std::endl;
            D_host(i) = 1.0f; // Avoid division by zero
        }
    }

    // Host buffer for convergence checking only
    Eigen::VectorXf residual_host(n);

    // ============================================================
    // SETUP PHASE: Upload everything to GPU ONCE
    // ============================================================

    // Device memory for vectors
    float *d_diag = nullptr, *d_x_old = nullptr, *d_x_new = nullptr;
    float *d_Ax = nullptr, *d_b = nullptr, *d_residual = nullptr;

    // Device memory for matrix A (CSR format)
    int *d_A_csrOffsets = nullptr, *d_A_columns = nullptr;
    float *d_A_values = nullptr;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_diag, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_x_old, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_x_new, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_Ax, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_b, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_residual, n * sizeof(float)));

    CUDA_CHECK(cudaMalloc((void**)&d_A_csrOffsets, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_A_columns, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_A_values, nnz * sizeof(float)));

    // Upload constant data ONCE
    CUDA_CHECK(cudaMemcpy(d_diag, D_host.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    // Upload matrix A in CSR format
    CUDA_CHECK(cudaMemcpy(d_A_csrOffsets, A.outerIndexPtr(),
                          (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A_columns, A.innerIndexPtr(),
                          nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A_values, A.valuePtr(),
                          nnz * sizeof(float), cudaMemcpyHostToDevice));

    // Initialize x = 0
    CUDA_CHECK(cudaMemset(d_x_old, 0, n * sizeof(float)));

    // ============================================================
    // CREATE CUSPARSE DESCRIPTORS (ONCE)
    // ============================================================

    cusparseHandle_t handle = nullptr;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecAx;

    CUSPARSE_CHECK(cusparseCreate(&handle));

    // Create sparse matrix descriptor for A
    CUSPARSE_CHECK(cusparseCreateCsr(&matA, n, n, nnz,
                                 d_A_csrOffsets, d_A_columns, d_A_values,
                                 CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                 CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    // Create dense vector descriptors
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecX, n, d_x_old, CUDA_R_32F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecAx, n, d_Ax, CUDA_R_32F));

    // Allocate SpMV buffer
    size_t bufferSize = 0;
    float alpha_tmp = 1.0f, beta_tmp = 0.0f;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha_tmp, matA, vecX, &beta_tmp, vecAx, CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));

    void* dBuffer = nullptr;
    CUDA_CHECK(cudaMalloc(&dBuffer, bufferSize));

    // Kernel launch parameters
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    std::cout << "Starting GPU A-Jacobi iterations (omega = " << omega << ")...\n";
    std::cout << "Block size: " << block_size << ", Grid size: " << grid_size << "\n";
    std::cout << "OPTIMIZED: All computation on GPU (no memcpy in loop)\n\n";

    bool converged = false;
    int final_iter = 0;

    // ============================================================
    // ITERATION LOOP: ALL ON GPU!
    // ============================================================

    for (int iter = 0; iter < max_iter; ++iter) {
        final_iter = iter + 1;

        // Step 1: Compute A*x entirely on GPU using cuSPARSE SpMV
        // Update vecX descriptor to point to current d_x_old
        CUSPARSE_CHECK(cusparseDnVecSetValues(vecX, d_x_old));

        // Compute d_Ax = A * d_x_old (entirely on GPU!)
        cuSparseSpMV_device(handle, matA, vecX, vecAx, 1.0f, 0.0f, dBuffer);

        // Step 2: Compute residual = A*x - b (on GPU)
        compute_residual_kernel<<<grid_size, block_size>>>(d_Ax, d_b, d_residual, n);
        CUDA_CHECK(cudaGetLastError());

        // Step 3: A-Jacobi update: x_new = x_old - omega * D^(-1) * residual (on GPU)
        a_jacobi_update_kernel<<<grid_size, block_size>>>(
            d_diag, d_residual, d_x_old, d_x_new, omega, n
        );
        CUDA_CHECK(cudaGetLastError());

        // Swap pointers
        float* temp = d_x_old;
        d_x_old = d_x_new;
        d_x_new = temp;

        // Check convergence every 10 iterations (only memcpy needed!)
        if (iter % 10 == 0 || iter == max_iter - 1) {
            CUDA_CHECK(cudaMemcpy(residual_host.data(), d_residual,
                                 n * sizeof(float), cudaMemcpyDeviceToHost));

            float residual_norm = residual_host.norm();
            std::cout << "Iteration " << final_iter << ": residual = " << residual_norm << "\n";

            if (residual_norm < tol) {
                std::cout << "\n✓ GPU A-Jacobi converged in " << final_iter << " iterations!\n";
                converged = true;
                break;
            }
        }
    }

    if (!converged) {
        std::cout << "\nGPU A-Jacobi reached max iterations (" << max_iter << ")\n";
    }

    // ============================================================
    // DOWNLOAD PHASE: Copy result back ONCE
    // ============================================================

    CUDA_CHECK(cudaMemcpy(x.data(), d_x_old, n * sizeof(float), cudaMemcpyDeviceToHost));

    // ============================================================
    // CLEANUP
    // ============================================================

    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecAx);
    cusparseDestroy(handle);

    cudaFree(dBuffer);
    cudaFree(d_diag);
    cudaFree(d_x_old);
    cudaFree(d_x_new);
    cudaFree(d_Ax);
    cudaFree(d_b);
    cudaFree(d_residual);
    cudaFree(d_A_csrOffsets);
    cudaFree(d_A_columns);
    cudaFree(d_A_values);
}

// -----------------------------------------------------------------
// CPU A-Jacobi Solver (for comparison)
// -----------------------------------------------------------------
void cpu_a_jacobi_solve(const Eigen::SparseMatrix<float, Eigen::RowMajor>& A,
                        const Eigen::VectorXf& b,
                        Eigen::VectorXf& x,
                        int max_iter,
                        float tol,
                        float omega = 0.67f)
{
    int n = A.rows();
    x.resize(n);
    x.setZero();

    Eigen::VectorXf x_new(n);
    Eigen::VectorXf D_inv(n);

    // Extract diagonal inverse
    for (int i = 0; i < n; ++i) {
        float diag = A.coeff(i, i);
        D_inv(i) = (diag != 0.0f) ? (1.0f / diag) : 0.0f;
    }

    std::cout << "Starting CPU A-Jacobi iterations (omega = " << omega << ")...\n\n";

    for (int iter = 0; iter < max_iter; ++iter) {
        // Compute residual: r = A*x - b
        Eigen::VectorXf residual = A * x - b;

        // A-Jacobi update: x_new = x - omega * D^(-1) * residual
        x_new = x - omega * (D_inv.asDiagonal() * residual);

        x = x_new;

        if (iter % 10 == 0 || iter == max_iter - 1) {
            float residual_norm = residual.norm();
            std::cout << "Iteration " << iter + 1 << ": residual = " << residual_norm << "\n";

            if (residual_norm < tol) {
                std::cout << "\n✓ CPU A-Jacobi converged in " << iter + 1 << " iterations!\n";
                return;
            }
        }
    }
    std::cout << "\nCPU A-Jacobi reached max iterations (" << max_iter << ")\n";
}

// -----------------------------------------------------------------
// Create a diagonally dominant SPD matrix
// -----------------------------------------------------------------
Eigen::SparseMatrix<float, Eigen::RowMajor> create_diag_dominant_matrix(int n) {
    std::vector<Eigen::Triplet<float>> triplets;
    for (int i = 0; i < n; ++i) {
        triplets.emplace_back(i, i, 10.0f + i * 0.1f);
    }
    for (int i = 0; i < n - 1; ++i) {
        float val = -0.5f;
        triplets.emplace_back(i, i+1, val);
        triplets.emplace_back(i+1, i, val);
    }
    for (int i = 0; i < n - 2; ++i) {
        float val = -0.2f;
        triplets.emplace_back(i, i+2, val);
        triplets.emplace_back(i+2, i, val);
    }
    Eigen::SparseMatrix<float, Eigen::RowMajor> A(n, n);
    A.setFromTriplets(triplets.begin(), triplets.end());
    A.makeCompressed();
    return A;
}

// -----------------------------------------------------------------
// Main
// -----------------------------------------------------------------
int main() {
    std::cout << "A-Jacobi Iterative Solver: CPU vs GPU (OPTIMIZED)\n";
    std::cout << "==================================================\n\n";

    // Check CUDA device
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!\n";
        return 1;
    }
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Using GPU: " << prop.name << "\n\n";

    // Parameters
    int max_iter = 200;
    float tol = 1e-6f;
    float omega = 0.67f; // Damping parameter

    // Test matrix
    std::cout << "========================================\n";
    std::cout << "Test (n=10000)\n";
    std::cout << "========================================\n";

    int n = 10000;
    Eigen::SparseMatrix<float, Eigen::RowMajor> A = create_diag_dominant_matrix(n);
    Eigen::VectorXf b = Eigen::VectorXf::Random(n);
    Eigen::VectorXf x_cpu(n);
    Eigen::VectorXf x_gpu(n);

    std::cout << "Matrix size: " << n << "x" << n << "\n";
    std::cout << "Non-zeros: " << A.nonZeros() << "\n";
    std::cout << "Omega (damping): " << omega << "\n\n";

    // CPU solve
    std::cout << "--- Running CPU A-Jacobi ---\n";
    auto t_cpu_start = std::chrono::high_resolution_clock::now();
    cpu_a_jacobi_solve(A, b, x_cpu, max_iter, tol, omega);
    auto t_cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(t_cpu_end - t_cpu_start).count();
    std::cout << "----------------------------\n";

    // GPU solve
    std::cout << "\n--- Running GPU A-Jacobi ---\n";
    auto t_gpu_start = std::chrono::high_resolution_clock::now();
    gpu_a_jacobi_solve(A, b, x_gpu, max_iter, tol, omega);
    auto t_gpu_end = std::chrono::high_resolution_clock::now();
    auto gpu_time = std::chrono::duration_cast<std::chrono::microseconds>(t_gpu_end - t_gpu_start).count();
    std::cout << "----------------------------\n";

    // Performance summary
    std::cout << "\n========================================\n";
    std::cout << "Performance Summary\n";
    std::cout << "========================================\n";
    std::cout << "CPU time: " << cpu_time << " μs ("
              << (float)cpu_time / 1000.0f << " ms)\n";
    std::cout << "GPU time: " << gpu_time << " μs ("
              << (float)gpu_time / 1000.0f << " ms)\n";
    std::cout << "Speedup: " << (float)cpu_time / gpu_time << "x\n\n";

    // Verification
    std::cout << "========================================\n";
    std::cout << "Verification\n";
    std::cout << "========================================\n";

    std::cout << std::fixed << std::setprecision(6);

    std::cout << "First 10 elements of x_cpu:\n[";
    for(int i = 0; i < 10; ++i) std::cout << x_cpu(i) << (i == 9 ? "" : ", ");
    std::cout << " ... ]\n";

    std::cout << "First 10 elements of x_gpu:\n[";
    for(int i = 0; i < 10; ++i) std::cout << x_gpu(i) << (i == 9 ? "" : ", ");
    std::cout << " ... ]\n\n";

    float error = (x_cpu - x_gpu).norm() / x_cpu.norm();
    Eigen::VectorXf residual_cpu = A * x_cpu - b;
    Eigen::VectorXf residual_gpu = A * x_gpu - b;

    std::cout << "Relative error ||x_cpu - x_gpu|| / ||x_cpu||: " << error << "\n";
    std::cout << "CPU residual ||A*x - b||: " << residual_cpu.norm() << "\n";
    std::cout << "GPU residual ||A*x - b||: " << residual_gpu.norm() << "\n";

    if (error < 1e-3f) {
        std::cout << "\n✓ Solutions match!\n";
    } else {
        std::cout << "\n✗ Solutions differ!\n";
    }

    return 0;
}
