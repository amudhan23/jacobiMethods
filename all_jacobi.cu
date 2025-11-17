#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <iomanip> // For std::setprecision

#include "cu_sparse.hpp" // Include our new SpGEMM helper

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
// R matrix (off-diagonal)
int *d_r_row_ptr = nullptr, *d_r_col_idx = nullptr;
float *d_r_values = nullptr;
// D vector (diagonal)
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
// KERNEL 1: A-Jacobi (Damped) Update
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
// KERNEL 2: Residual Kernel (for Damped & Standard Jacobi)
// -----------------------------------------------------------------
__global__ void compute_residual_kernel_v2(
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
// KERNEL 3: Standard Jacobi (Paper's "l=1 A-Jacobi")
// -----------------------------------------------------------------
__global__ void standard_jacobi_kernel(
    const int* r_row_ptr,    // R matrix row pointers
    const int* r_col_idx,    // R matrix column indices
    const float* r_values,   // R matrix values (off-diagonal)
    const float* diag,       // D vector (a_ii values)
    const float* b,          // Right-hand side
    const float* x_old,      // Previous solution
    float* x_new,            // New solution (output)
    int n)                   // Matrix size
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        float sum_Rx = 0.0f; // This is the R*x_old part

        int row_start = r_row_ptr[i];
        int row_end = r_row_ptr[i + 1];

        // This loop calculates the sum over off-diagonals
        for (int j = row_start; j < row_end; ++j) {
            sum_Rx += r_values[j] * x_old[r_col_idx[j]];
        }

        float diag_val = diag[i]; 

        // Jacobi update: x_new[i] = (b[i] - sum_Rx) / diag_val
        if (diag_val != 0.0f) {
            x_new[i] = (b[i] - sum_Rx) / diag_val;
        } else {
            x_new[i] = 0.0f; // Or some other handling
        }
    }
}

// -----------------------------------------------------------------
// KERNEL 4 (NEW): Polynomial Jacobi (l=2)
// x_k = D_term + S^2 * x_k-2
// -----------------------------------------------------------------
__global__ void polynomial_jacobi_l2_kernel(
    const int* s2_row_ptr,      // S^2 matrix row pointers
    const int* s2_col_idx,      // S^2 matrix column indices
    const float* s2_values,     // S^2 matrix values
    const float* d_term,        // D_term vector (I+S)D^-1 * b
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
// setup_gpu_solver (MODIFIED)
// Now computes and uploads R, D, S^2, and D_term
// -----------------------------------------------------------------
void setup_gpu_solver(const Eigen::SparseMatrix<float, Eigen::RowMajor>& A,
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
    // This calls our cuSPARSE SpGEMM code
    Eigen::SparseMatrix<float, Eigen::RowMajor> S2_host = compute_S_squared(A);
    g_nnz_s2 = S2_host.nonZeros();

    // --- Compute D_term = (I + S) * (D^-1 * b) ---
    // S = D^-1 * R
    // D_term = D^-1 * b + (D^-1 * R) * (D^-1 * b)
    printf("Computing polynomial D_term on CPU...\n");
    Eigen::VectorXf D_inv_b = D_inv_host.asDiagonal() * b_host;
    Eigen::SparseMatrix<float, Eigen::RowMajor> S_host = R_host;
    for (int k=0; k<S_host.outerSize(); ++k) {
        for (Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(S_host,k); it; ++it) {
            it.valueRef() *= D_inv_host(it.row());
        }
    }
    Eigen::VectorXf D_term_host = D_inv_b + (S_host * D_inv_b);
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

    // Allocate new memory for S2 and D_term
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
// cleanup_gpu_solver (MODIFIED)
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
    // Free new pointers
    cudaFree(d_s2_row_ptr);
    cudaFree(d_s2_col_idx);
    cudaFree(d_s2_values);
    cudaFree(d_d_term);
}

// -----------------------------------------------------------------
// SOLVER 1: gpu_a_jacobi_solve (Damped Jacobi)
// -----------------------------------------------------------------
void gpu_a_jacobi_solve(const Eigen::VectorXf& D_host, // Pass host diagonal
                        const Eigen::VectorXf& b_host, // Pass host RHS
                        Eigen::VectorXf& x_host,       // Host output vector
                        int max_iter,
                        float tol,
                        float omega = 0.67f) { // Damping parameter
    
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
    CUDA_CHECK(cudaMemset(d_x_old, 0, n * sizeof(float)));

    // Kernel launch parameters
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    std::cout << "Starting GPU A-Jacobi iterations (omega = " << omega << ")...\n";
    std::cout << "Block size: " << block_size << ", Grid size: " << grid_size << "\n\n";

    // Allocate host memory for checking convergence
    std::vector<float> h_residual(n);
    bool converged = false;
    int final_iter = 0;
    
    for (int iter = 0; iter < max_iter; ++iter) {
        final_iter = iter + 1;

        // Step 1: Compute residual r = A*x_old - b
        compute_residual_kernel_v2<<<grid_size, block_size>>>(
            d_r_row_ptr, d_r_col_idx, d_r_values, d_diag,
            d_x_old, // Use current x
            d_b, d_residual, n
        );
        CUDA_CHECK(cudaGetLastError());
        
        // Step 2: Check convergence
        if (iter % 10 == 0 || iter == max_iter - 1) {
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
                std::cout << "\n✓ GPU Damped converged in " << final_iter << " iterations!\n";
                converged = true;
                break;
            }
        }
        
        // Step 3: A-Jacobi update: x_new = x_old - omega * D^(-1) * r
        a_jacobi_update_kernel<<<grid_size, block_size>>>(
            d_diag, d_residual, d_x_old, d_x_new, omega, n
        );
        CUDA_CHECK(cudaGetLastError());

        // Step 4: Swap x_old and x_new for next iteration
        float* temp = d_x_old;
        d_x_old = d_x_new;
        d_x_new = temp;
    }
    
    if (!converged) {
         std::cout << "\nGPU Damped reached max iterations (" << max_iter << ")\n";
    }

    // --- Copy solution back to host ---
    // The final, correct solution is now always in d_x_old (due to the swap)
    std::cout << "Transferring solution from GPU to CPU...\n";
    CUDA_CHECK(cudaMemcpy(x_host.data(), d_x_old, n * sizeof(float), cudaMemcpyDeviceToHost));
}


// -----------------------------------------------------------------
// SOLVER 2: gpu_standard_jacobi_solve (Paper's l=1 A-Jacobi)
// -----------------------------------------------------------------
void gpu_standard_jacobi_solve(const Eigen::VectorXf& D_host, // Pass host diagonal
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
    CUDA_CHECK(cudaMemset(d_x_old, 0, n * sizeof(float)));

    // Kernel launch parameters
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    std::cout << "Starting GPU Standard Jacobi (l=1 Aggregated Jacobi) iterations...\n";
    std::cout << "Block size: " << block_size << ", Grid size: " << grid_size << "\n\n";

    // Allocate host memory for checking convergence
    std::vector<float> h_residual(n);
    bool converged = false;
    int final_iter = 0;
    
    for (int iter = 0; iter < max_iter; ++iter) {
        final_iter = iter + 1;

        // Step 1: Compute x_new = D^-1 * (b - R*x_old)
        standard_jacobi_kernel<<<grid_size, block_size>>>(
            d_r_row_ptr, d_r_col_idx, d_r_values,
            d_diag, d_b, d_x_old, d_x_new, n
        );
        CUDA_CHECK(cudaGetLastError());
        
        // Step 2: Swap x_old and x_new *before* checking
        // The most up-to-date solution is now in d_x_old
        float* temp = d_x_old;
        d_x_old = d_x_new;
        d_x_new = temp;

        // Step 3: Check convergence
        if (iter % 10 == 0 || iter == max_iter - 1) {
            // Compute residual r = A*x_old - b
            compute_residual_kernel_v2<<<grid_size, block_size>>>(
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
                std::cout << "\n✓ GPU Standard (l=1) converged in " << final_iter << " iterations!\n";
                converged = true;
                break;
            }
        }
    }
    
    if (!converged) {
         std::cout << "\nGPU Standard (l=1) reached max iterations (" << max_iter << ")\n";
    }

    // --- Copy solution back to host ---
    // The final, correct solution is now always in d_x_old
    std::cout << "Transferring solution from GPU to CPU...\n";
    CUDA_CHECK(cudaMemcpy(x_host.data(), d_x_old, n * sizeof(float), cudaMemcpyDeviceToHost));
}

// -----------------------------------------------------------------
// SOLVER 3 (NEW): gpu_polynomial_jacobi_l2_solve
// Implements x_k = D_term + S^2 * x_k-2
// -----------------------------------------------------------------
void gpu_polynomial_jacobi_l2_solve(const Eigen::VectorXf& D_host, // Pass host diagonal
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
    // We only need to upload D and b if they change.
    // For this solver, D_term was already precomputed from b,
    // so we only need to upload D (for the residual check)
    CUDA_CHECK(cudaMemcpy(d_diag, D_host.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b_host.data(), n * sizeof(float), cudaMemcpyHostToDevice));


    // Initialize x = 0
    CUDA_CHECK(cudaMemset(d_x_old, 0, n * sizeof(float))); // x_k-2
    CUDA_CHECK(cudaMemset(d_x_new, 0, n * sizeof(float))); // x_k

    // Kernel launch parameters
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    std::cout << "Starting GPU Polynomial Jacobi (l=2) iterations...\n";
    std::cout << "Block size: " << block_size << ", Grid size: " << grid_size << "\n\n";

    // Allocate host memory for checking convergence
    std::vector<float> h_residual(n);
    bool converged = false;
    int final_iter = 0;
    
    // This is a two-step iteration, so we read from x_old, write to x_new
    // Then swap. We must "prime" the loop by computing x_1 first.
    
    // Compute x_1 (using l=1 kernel)
    standard_jacobi_kernel<<<grid_size, block_size>>>(
        d_r_row_ptr, d_r_col_idx, d_r_values,
        d_diag, d_b, d_x_old, d_x_new, n
    );
    // Now d_x_old = x_0 (zeros) and d_x_new = x_1
    float* temp_prime = d_x_old;
    d_x_old = d_x_new; // d_x_old = x_1
    d_x_new = temp_prime; // d_x_new = x_0
    
    // l=2 loop computes 2 steps at a time, so max_iter/2
    for (int iter = 1; iter < max_iter; iter += 2) { // Note: iter starts at 1, increments by 2
        final_iter = iter + 1; // We compute 2 steps

        // Step 1: Compute x_k = D_term + S^2 * x_k-2
        // d_x_old has x_k-2 (e.g., x_1)
        // d_x_new will get x_k   (e.g., x_3)
        polynomial_jacobi_l2_kernel<<<grid_size, block_size>>>(
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
        if (iter % 10 == 1 || iter >= max_iter - 2) { // Check ~every 10 iters
            // Compute residual r = A*x_old - b
            compute_residual_kernel_v2<<<grid_size, block_size>>>(
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
                std::cout << "\n✓ GPU Polynomial (l=2) converged in " << final_iter << " iterations!\n";
                converged = true;
                break;
            }
        }
    }
    
    if (!converged) {
         std::cout << "\nGPU Polynomial (l=2) reached max iterations (" << max_iter << ")\n";
    }

    // --- Copy solution back to host ---
    // The final, correct solution is now always in d_x_old
    std::cout << "Transferring solution from GPU to CPU...\n";
    CUDA_CHECK(cudaMemcpy(x_host.data(), d_x_old, n * sizeof(float), cudaMemcpyDeviceToHost));
}


// -----------------------------------------------------------------
// SOLVER 4: cpu_a_jacobi_solve (Damped Jacobi)
// -----------------------------------------------------------------
void cpu_a_jacobi_solve(const Eigen::SparseMatrix<float, Eigen::RowMajor>& A,
                        const Eigen::VectorXf& b,
                        Eigen::VectorXf& x,
                        int max_iter,
                        float tol,
                        float omega = 0.67f) { // Damping parameter
    int n = A.rows();
    Eigen::VectorXf x_new(n);
    x.setZero();

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

        x = x_new; // Update x for next iteration

        if (iter % 10 == 0 || iter == max_iter - 1) {
            float res_norm = residual.norm();
            std::cout << "Iteration " << iter+1 << ": residual = " << res_norm << "\n";
            if (res_norm < tol) {
                std::cout << "\n✓ CPU A-Jacobi converged in " << iter + 1 << " iterations!\n";
                return;
            }
        }
    }
    std::cout << "\nCPU A-Jacobi reached max iterations (" << max_iter << ")\n";
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

// -----------------------------------------------------------------
// MODIFIED: main()
// Now calls all solvers
// -----------------------------------------------------------------
int main() {
    std::cout << "Jacobi Iterative Solver: CPU vs GPU (Optimized)\n";
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
    float omega = 0.67f; // Damping factor

    // --- Large test ---
    std::cout << "========================================\n";
    std::cout << "Large Test (n=100000)\n";
    std::cout << "========================================\n";

    int n_large = 100000;
    Eigen::SparseMatrix<float, Eigen::RowMajor> A_large = create_diag_dominant_matrix(n_large);
    Eigen::VectorXf b_large = Eigen::VectorXf::Random(n_large);
    Eigen::VectorXf x_cpu_large(n_large);
    Eigen::VectorXf x_gpu_damped_large(n_large);
    Eigen::VectorXf x_gpu_standard_large(n_large); 
    Eigen::VectorXf x_gpu_polynomial_l2_large(n_large); // For the new l=2 solver
    
    // This will hold the diagonal elements
    Eigen::VectorXf D_host_large; 

    std::cout << "Matrix size: " << n_large << "x" << n_large << "\n";
    std::cout << "Non-zeros: " << A_large.nonZeros() << "\n\n";

    // --- CPU solve ---
    std::cout << "--- Running CPU A-Jacobi ---\n";
    auto t_cpu_large_start = std::chrono::high_resolution_clock::now();
    cpu_a_jacobi_solve(A_large, b_large, x_cpu_large, max_iter, tol, omega);
    auto t_cpu_large_end = std::chrono::high_resolution_clock::now();
    auto cpu_large_time = std::chrono::duration_cast<std::chrono::microseconds>(t_cpu_large_end - t_cpu_large_start).count();
    std::cout << "----------------------------\n";


    // --- GPU solve (A-Jacobi / Damped) ---
    std::cout << "\n--- Running GPU A-Jacobi (Damped) ---\n";
    // 1. One-time setup
    auto t_setup_start = std::chrono::high_resolution_clock::now();
    // Setup now needs b_host to precompute D_term
    setup_gpu_solver(A_large, b_host, D_host_large);
    auto t_setup_end = std::chrono::high_resolution_clock::now();
    auto setup_time = std::chrono::duration_cast<std::chrono::microseconds>(t_setup_end - t_setup_start).count();
    std::cout << "GPU Setup Time: " << setup_time << " us\n";

    // 2. Run the solver
    auto t_gpu_large_start = std::chrono::high_resolution_clock::now();
    gpu_a_jacobi_solve(D_host_large, b_large, x_gpu_damped_large, max_iter, tol, omega);
    auto t_gpu_large_end = std::chrono::high_resolution_clock::now();
    auto gpu_damped_time = std::chrono::duration_cast<std::chrono::microseconds>(t_gpu_large_end - t_gpu_large_start).count();

    // --- GPU solve (Standard Jacobi) ---
    std::cout << "\n--- Running GPU Standard Jacobi (l=1 Aggregated) ---\n";
    // We re-use the setup, but not the cleanup
    auto t_gpu_std_start = std::chrono::high_resolution_clock::now();
    gpu_standard_jacobi_solve(D_host_large, b_large, x_gpu_standard_large, max_iter, tol);
    auto t_gpu_std_end = std::chrono::high_resolution_clock::now();
    auto gpu_standard_time = std::chrono::duration_cast<std::chrono::microseconds>(t_gpu_std_end - t_gpu_std_start).count();

    // --- GPU solve (Polynomial Jacobi l=2) ---
    std::cout << "\n--- Running GPU Polynomial Jacobi (l=2) ---\n";
    // We re-use the setup, but not the cleanup
    auto t_gpu_poly_start = std::chrono::high_resolution_clock::now();
    gpu_polynomial_jacobi_l2_solve(D_host_large, b_large, x_gpu_polynomial_l2_large, max_iter, tol);
    auto t_gpu_poly_end = std::chrono::high_resolution_clock::now();
    auto gpu_polynomial_time = std::chrono::duration_cast<std::chrono::microseconds>(t_gpu_poly_end - t_gpu_poly_start).count();


    // 3. One-time cleanup
    cleanup_gpu_solver();
    std::cout << "---------------------------\n";


    // --- Performance Summary ---
    std::cout << "\n========================================\n";
    std::cout << "Performance Summary\n";
    std::cout << "========================================\n";
    std::cout << "CPU time: " << cpu_large_time << " μs (" 
              << (float)cpu_large_time / 1000.0f << " ms)\n";
    std::cout << "GPU (Damped) time: " << gpu_damped_time << " μs (" 
              << (float)gpu_damped_time / 1000.0f << " ms)\n";
    std::cout << "GPU (Standard l=1) time: " << gpu_standard_time << " μs (" 
              << (float)gpu_standard_time / 1000.0f << " ms)\n";
    std::cout << "GPU (Polynomial l=2) time: " << gpu_polynomial_time << " μs (" 
              << (float)gpu_polynomial_time / 1000.0f << " ms)\n";
    std::cout << "GPU Setup time: " << setup_time << " μs (" 
              << (float)setup_time / 1000.0f << " ms)\n";
    
    // Guard against division by zero if GPU is too fast
    if (gpu_damped_time > 0) {
        std::cout << "Speedup (Damped vs CPU): " << (float)cpu_large_time / gpu_damped_time << "x\n";
    }
    if (gpu_standard_time > 0) {
        std::cout << "Speedup (Standard l=1 vs CPU): " << (float)cpu_large_time / gpu_standard_time << "x\n";
    }
    if (gpu_polynomial_time > 0) {
        std::cout << "Speedup (Polynomial l=2 vs CPU): " << (float)cpu_large_time / gpu_polynomial_time << "x\n\n";
    }

    // --- Verification & Displaying Solution ---
    std::cout << "========================================\n";
    std::cout << "Verification\n";
    std::cout << "========================================\n";
    
    // Set precision for printing vectors
    std::cout << std::fixed << std::setprecision(6);
    
    std::cout << "First 10 elements of x_cpu (Damped):\n[";
    for(int i = 0; i < 10; ++i) std::cout << x_cpu_large(i) << (i == 9 ? "" : ", ");
    std::cout << " ... ]\n";
    
    std::cout << "First 10 elements of x_gpu (Damped):\n[";
    for(int i = 0; i < 10; ++i) std::cout << x_gpu_damped_large(i) << (i == 9 ? "" : ", ");
    std::cout << " ... ]\n";

    std::cout << "First 10 elements of x_gpu (Standard l=1):\n[";
    for(int i = 0; i < 10; ++i) std::cout << x_gpu_standard_large(i) << (i == 9 ? "" : ", ");
    std::cout << " ... ]\n";

    std::cout << "First 10 elements of x_gpu (Polynomial l=2):\n[";
    for(int i = 0; i < 10; ++i) std::cout << x_gpu_polynomial_l2_large(i) << (i == 9 ? "" : ", ");
    std::cout << " ... ]\n\n";
    
    float error_damped = (x_cpu_large - x_gpu_damped_large).norm() / x_cpu_large.norm();
    float error_standard = (x_cpu_large - x_gpu_standard_large).norm(); // Compare to CPU
    float error_polynomial = (x_cpu_large - x_gpu_polynomial_l2_large).norm();
    
    Eigen::VectorXf residual_gpu_damped = A_large * x_gpu_damped_large - b_large;
    Eigen::VectorXf residual_gpu_standard = A_large * x_gpu_standard_large - b_large;
    Eigen::VectorXf residual_gpu_polynomial = A_large * x_gpu_polynomial_l2_large - b_large;


    std::cout << "Relative error ||x_cpu - x_gpu_damped|| / ||x_cpu||: " << error_damped << "\n";
    std::cout << "Absolute error ||x_cpu - x_gpu_standard||: " << error_standard << "\n";
    std::cout << "Absolute error ||x_cpu - x_gpu_polynomial||: " << error_polynomial << "\n\n";
    
    std::cout << "CPU (Damped) final residual ||A*x - b||: " << (A_large * x_cpu_large - b_large).norm() << "\n";
    std::cout << "GPU (Damped) final residual ||A*x - b||: " << residual_gpu_damped.norm() << "\n";
    std::cout << "GPU (Standard l=1) final residual ||A*x - b||: " << residual_gpu_standard.norm() << "\n";
    std::cout << "GPU (Polynomial l=2) final residual ||A*x - b||: " << residual_gpu_polynomial.norm() << "\n";


    if (error_damped < 1e-3f) {
        std::cout << "\n✓ Damped solutions match!\n";
    }

    return 0;
}