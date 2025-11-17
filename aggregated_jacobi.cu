#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <iomanip> // For std::setprecision

#include "cu_sparse.hpp" // Include our SpGEMM helper

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

// --- Device Pointers (Global for solver state) ---
// R matrix (off-diagonal) - needed for residual check
int *d_r_row_ptr = nullptr, *d_r_col_idx = nullptr;
float *d_r_values = nullptr;
// D vector (diagonal) - needed for residual check
float *d_diag = nullptr;
// B vector (RHS)
float *d_b = nullptr;
// S^2 matrix (for polynomial solver)
int *d_s2_row_ptr = nullptr, *d_s2_col_idx = nullptr;
float *d_s2_values = nullptr;
// D_term vector (for polynomial solver)
float *d_d_term = nullptr;

// Solution vectors
float *d_x_old = nullptr, *d_x_new = nullptr, *d_residual = nullptr;
int g_n = 0;
int g_nnz_r = 0;
int g_nnz_s2 = 0;


// -----------------------------------------------------------------
// KERNEL 1: Residual Kernel (Used for convergence check)
// Computes residual as (R+D)*x - b
// -----------------------------------------------------------------
__global__ void compute_residual_kernel(
    const int* r_row_ptr,
    const int* r_col_idx,
    const float* r_values,
    const float* diag,
    const float* x,
    const float* b,
    float* residual,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        float Ax_i = 0.0f;

        // 1. Add the R*x part (from R matrix)
        int row_start = r_row_ptr[i];
        int row_end = r_row_ptr[i + 1];
        for (int j = row_start; j < row_end; ++j) {
            Ax_i += r_values[j] * x[r_col_idx[j]];
        }

        // 2. Add the D*x part (from diag vector)
        Ax_i += diag[i] * x[i]; 

        // 3. Compute residual
        residual[i] = Ax_i - b[i];
    }
}

// -----------------------------------------------------------------
// KERNEL 2: Aggregated (Polynomial) Jacobi (l=2)
// x_k = D_term + S^2 * x_k-2
// -----------------------------------------------------------------
__global__ void aggregated_jacobi_l2_kernel(
    const int* s2_row_ptr,      // S^2 matrix row pointers
    const int* s2_col_idx,      // S^2 matrix column indices
    const float* s2_values,     // S^2 matrix values
    const float* d_term,        // D_term vector (I-S)D^-1 * b
    const float* x_k_minus_2,   // Solution from two steps ago
    float* x_k,                 // New solution (output)
    int n)                      // Matrix size
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Current row (k)

    if (i < n) {
        float sum_S2x = 0.0f; // This will be S^2 * x_k-2

        int row_start = s2_row_ptr[i];
        int row_end = s2_row_ptr[i + 1];

        // This loop computes S^2 * x_k-2
        for (int j_idx = row_start; j_idx < row_end; ++j_idx) {
            sum_S2x += s2_values[j_idx] * x_k_minus_2[s2_col_idx[j_idx]];
        }

        // Final update: x_k = d_term[i] + sum_S2x
        x_k[i] = d_term[i] + sum_S2x;
    }
}


// -----------------------------------------------------------------
// setup_gpu_solver
// Computes and uploads R, D, S^2, and D_term
// -----------------------------------------------------------------
void setup_gpu_solver(const Eigen::SparseMatrix<float, Eigen::RowMajor>& A,
    Eigen::SparseMatrix<float, Eigen::RowMajor> S2_host,
                      const Eigen::VectorXf& b_host,
                      Eigen::VectorXf& D_host) 
{
    g_n = A.rows();
    std::vector<Eigen::Triplet<float>> r_triplets;
    D_host.resize(g_n);
    Eigen::VectorXf D_inv_host(g_n);

    std::cout << "Splitting A into D (diagonal) and R (off-diagonal)...\n";
    for (int k = 0; k < A.outerSize(); ++k) {
        for (Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(A, k); it; ++it) {
            if (it.row() == it.col()) {
                D_host(it.row()) = it.value();
                D_inv_host(it.row()) = 1.0f / it.value();
            } else {
                r_triplets.emplace_back(it.row(), it.col(), it.value());
            }
        }
    }

    Eigen::SparseMatrix<float, Eigen::RowMajor> R_host(g_n, g_n);
    R_host.setFromTriplets(r_triplets.begin(), r_triplets.end());
    R_host.makeCompressed();

    g_nnz_r = R_host.nonZeros();
    std::cout << "R matrix non-zeros: " << g_nnz_r << "\n";

    // --- Compute S^2 ---
    // This calls our cuSPARSE SpGEMM code in cu_sparse.cu
    g_nnz_s2 = S2_host.nonZeros();

    // --- Compute D_term = (I - S) * (D^-1 * b) ---
    // S = D^-1 * R
    // D_term = D^-1 * b - (D^-1 * R) * (D^-1 * b)
    printf("Computing polynomial D_term on CPU...\n");
    Eigen::VectorXf D_inv_b = D_inv_host.asDiagonal() * b_host; // c = D^-1 * b
    Eigen::SparseMatrix<float, Eigen::RowMajor> S_host = R_host;
    for (int k=0; k<S_host.outerSize(); ++k) {
        for (Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(S_host,k); it; ++it) {
            it.valueRef() *= D_inv_host(it.row()); // S_host is now S = D^-1 * R
        }
    }
    // *** THIS IS THE FIX ***
    // We compute c - S*c, which is (I - S)c
    Eigen::VectorXf D_term_host = D_inv_b - (S_host * D_inv_b);
    printf("D_term computed.\n\n");


    // --- Allocate ALL GPU memory ---
    CUDA_CHECK(cudaMalloc(&d_r_row_ptr, (g_n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_r_col_idx, g_nnz_r * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_r_values, g_nnz_r * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_diag, g_n * sizeof(float))); 
    CUDA_CHECK(cudaMalloc(&d_b, g_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x_old, g_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x_new, g_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_residual, g_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_s2_row_ptr, (g_n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_s2_col_idx, g_nnz_s2 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_s2_values, g_nnz_s2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_d_term, g_n * sizeof(float)));

    // --- Perform ONE-TIME upload for unchanging data ---
    std::cout << "Uploading R (off-diagonals) to GPU...\n";
    CUDA_CHECK(cudaMemcpy(d_r_row_ptr, R_host.outerIndexPtr(), (g_n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_r_col_idx, R_host.innerIndexPtr(), g_nnz_r * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_r_values, R_host.valuePtr(), g_nnz_r * sizeof(float), cudaMemcpyHostToDevice));

    std::cout << "Uploading S^2 (polynomial matrix) to GPU...\n";
    CUDA_CHECK(cudaMemcpy(d_s2_row_ptr, S2_host.outerIndexPtr(), (g_n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_s2_col_idx, S2_host.innerIndexPtr(), g_nnz_s2 * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_s2_values, S2_host.valuePtr(), g_nnz_s2 * sizeof(float), cudaMemcpyHostToDevice));

    std::cout << "Uploading D_term (polynomial vector) to GPU...\n";
    CUDA_CHECK(cudaMemcpy(d_d_term, D_term_host.data(), g_n * sizeof(float), cudaMemcpyHostToDevice));
}

// -----------------------------------------------------------------
// cleanup_gpu_solver
// -----------------------------------------------------------------
void cleanup_gpu_solver() {
    std::cout << "\nCleaning up GPU solver memory...\n";
    cudaFree(d_r_row_ptr);
    cudaFree(d_r_col_idx);
    cudaFree(d_r_values);
    cudaFree(d_diag);
    cudaFree(d_b);
    cudaFree(d_x_old);
    cudaFree(d_x_new);
    cudaFree(d_residual);
    cudaFree(d_s2_row_ptr);
    cudaFree(d_s2_col_idx);
    cudaFree(d_s2_values);
    cudaFree(d_d_term);
}

// -----------------------------------------------------------------
// SOLVER: gpu_aggregated_jacobi_solve (l=2)
// Implements x_k = D_term + S^2 * x_k-2
// -----------------------------------------------------------------
void gpu_aggregated_jacobi_solve(const Eigen::VectorXf& D_host, // Pass host diagonal
                                    const Eigen::VectorXf& b_host, // Pass host RHS
                                    Eigen::VectorXf& x_host,       // Host output vector
                                    int max_iter,
                                    float tol) {
    
    int n = b_host.size();
    if (n != g_n) {
        std::cerr << "Error: Solver setup for n=" << g_n 
                  << " but called with n=" << n << std::endl;
        return;
    }

    // --- FAST UPLOADS (per-frame) ---
    CUDA_CHECK(cudaMemcpy(d_diag, D_host.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b_host.data(), n * sizeof(float), cudaMemcpyHostToDevice));


    // Initialize x = 0
    CUDA_CHECK(cudaMemset(d_x_old, 0, n * sizeof(float))); // x_k-2
    CUDA_CHECK(cudaMemset(d_x_new, 0, n * sizeof(float))); // x_k

    // Kernel launch parameters
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    std::cout << "Starting GPU Aggregated (Polynomial l=2) iterations...\n";
    std::cout << "Block size: " << block_size << ", Grid size: " << grid_size << "\n\n";

    // Allocate host memory for checking convergence
    std::vector<float> h_residual(n);
    bool converged = false;
    int final_iter = 0;
    
    // This is a two-step iteration (k depends on k-2).
    // We will compute the even iterations (x_0 -> x_2, x_2 -> x_4, etc.)
    // d_x_old will hold x_k-2
    // d_x_new will hold x_k
    
    // l=2 loop computes 2 steps at a time, so max_iter/2
    for (int iter = 0; iter < max_iter; iter += 2) { // Note: iter starts at 0, increments by 2
        final_iter = iter + 2; // We are computing 2 steps ahead

        // Step 1: Compute x_k = D_term + S^2 * x_k-2
        // d_x_old has x_k-2 (e.g., x_0 on first iter)
        // d_x_new will get x_k   (e.g., x_2 on first iter)
        aggregated_jacobi_l2_kernel<<<grid_size, block_size>>>(
            d_s2_row_ptr, d_s2_col_idx, d_s2_values,
            d_d_term, d_x_old, d_x_new, n
        );
        CUDA_CHECK(cudaGetLastError());
        
        // Step 2: Swap x_old and x_new *before* checking
        // The most up-to-date solution (x_k) is now in d_x_old
        float* temp = d_x_old;
        d_x_old = d_x_new;
        d_x_new = temp;

        // Step 3: Check convergence
        if (iter % 10 == 0 || iter >= max_iter - 2) { // Check ~every 10 iters
            // Compute residual r = A*x_old - b
            compute_residual_kernel<<<grid_size, block_size>>>(
                d_r_row_ptr, d_r_col_idx, d_r_values, d_diag,
                d_x_old, // Check the solution we just computed
                d_b, d_residual, n
            );
            CUDA_CHECK(cudaGetLastError());

            // Copy residual to host
            CUDA_CHECK(cudaMemcpy(h_residual.data(), d_residual, n * sizeof(float), cudaMemcpyDeviceToHost));

            // Compute norm
            float residual_norm = 0.0f;
            for (int i = 0; i < n; ++i) {
                residual_norm += h_residual[i] * h_residual[i];
            }
            residual_norm = std::sqrt(residual_norm);

            std::cout << "Iteration " << final_iter << ": residual = " << residual_norm << "\n";

            if (residual_norm < tol) {
                std::cout << "\n✓ GPU Aggregated (l=2) converged in " << final_iter << " iterations!\n";
                converged = true;
                break;
            }
        }
    }
    
    if (!converged) {
         std::cout << "\nGPU Aggregated (l=2) reached max iterations (" << max_iter << ")\n";
    }

    // --- Copy solution back to host ---
    // The final, correct solution is now always in d_x_old
    std::cout << "Transferring solution from GPU to CPU...\n";
    CUDA_CHECK(cudaMemcpy(x_host.data(), d_x_old, n * sizeof(float), cudaMemcpyDeviceToHost));
}


// Create a diagonally dominant SPD matrix (Unchanged)
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


Eigen::VectorXf solve_with_l2_jacobi(
    const Eigen::SparseMatrix<float, Eigen::RowMajor>& A,
    Eigen::SparseMatrix<float, Eigen::RowMajor> S2_host,
    const Eigen::VectorXf& b,
    int max_iter,
    float tol) 
{
    std::cout << "\n--- Running GPU Aggregated (Polynomial l=2) ---\n";
    
    Eigen::VectorXf x_solution(A.rows());
    Eigen::VectorXf D_host; // Will be filled by setup

    // 1. One-time setup
    auto t_setup_start = std::chrono::high_resolution_clock::now();
    // Setup now needs b_host to precompute D_term
    setup_gpu_solver(A, S2_host, b, D_host);
    auto t_setup_end = std::chrono::high_resolution_clock::now();
    auto setup_time = std::chrono::duration_cast<std::chrono::microseconds>(t_setup_end - t_setup_start).count();
    std::cout << "GPU Setup Time: " << (float)setup_time / 1000.0f << " ms\n";

    // 2. Run the solver
    auto t_gpu_large_start = std::chrono::high_resolution_clock::now();
    gpu_aggregated_jacobi_solve(D_host, b, x_solution, max_iter, tol);
    auto t_gpu_large_end = std::chrono::high_resolution_clock::now();
    auto gpu_solve_time = std::chrono::duration_cast<std::chrono::microseconds>(t_gpu_large_end - t_gpu_large_start).count();
    std::cout << "GPU Solve Time: " << (float)gpu_solve_time / 1000.0f << " ms\n";

    // 3. One-time cleanup
    cleanup_gpu_solver();
    std::cout << "---------------------------\n";

    // Print stats from within the wrapper
    std::cout << "\n--- Solver Stats ---\n";
    std::cout << "GPU Setup time (includes SpGEMM): " << (float)setup_time / 1000.0f << " ms\n";
    std::cout << "GPU Solve time: " << (float)gpu_solve_time / 1000.0f << " ms\n";
    std::cout << "Total GPU time: " << (float)(setup_time + gpu_solve_time) / 1000.0f << " ms\n\n";

    return x_solution;
}


// -----------------------------------------------------------------
// main() - Commented out because main.cpp has the main function
// -----------------------------------------------------------------
/*
int main() {
    std::cout << "Aggregated (Polynomial l=2) Jacobi Solver\n";
    std::cout << "================================================\n\n";

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

    // --- Shared Parameters ---
    int max_iter = 100;
    float tol = 1e-6f;

    // --- Large test ---
    std::cout << "========================================\n";
    std::cout << "Large Test (n=100000)\n";
    std::cout << "========================================\n";

    int n_large = 100000;
    Eigen::SparseMatrix<float, Eigen::RowMajor> A_large = create_diag_dominant_matrix(n_large);
    Eigen::VectorXf b_large = Eigen::VectorXf::Random(n_large);

    std::cout << "Matrix size: " << n_large << "x" << n_large << "\n";
    std::cout << "Non-zeros: " << A_large.nonZeros() << "\n\n";

    // Compute S2 = A * A using cuSparseMultiply
    Eigen::SparseMatrix<float, Eigen::RowMajor> S2_large = cuSparseMultiply(A_large);

    // --- Run the new wrapper function ---
    Eigen::VectorXf x_gpu_large = solve_with_l2_jacobi(A_large, S2_large, b_large, max_iter, tol);


    // --- Verification & Displaying Solution ---
    std::cout << "========================================\n";
    std::cout << "Verification\n";
    std::cout << "========================================\n";

    // Set precision for printing vectors
    std::cout << std::fixed << std::setprecision(6);

    std::cout << "First 10 elements of x_gpu (Aggregated l=2):\n[";
    for(int i = 0; i < 10; ++i) std::cout << x_gpu_large(i) << (i == 9 ? "" : ", ");
    std::cout << " ... ]\n\n";

    Eigen::VectorXf residual_gpu = A_large * x_gpu_large - b_large;

    std::cout << "GPU (Aggregated l=2) final residual ||A*x - b||: " << residual_gpu.norm() << "\n";

    if (residual_gpu.norm() < tol * 10) { // Give a little slack vs. internal check
        std::cout << "\n✓ Solution converged!\n";
    }

    return 0;
}
*/