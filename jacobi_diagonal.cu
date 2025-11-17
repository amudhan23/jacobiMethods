#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <iomanip> // For std::setprecision

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
// We make these global so setup and cleanup are separate.
// In a real class, these would be member variables.
int *d_r_row_ptr = nullptr, *d_r_col_idx = nullptr;
float *d_r_values = nullptr, *d_diag = nullptr, *d_b = nullptr;
float *d_x_old = nullptr, *d_x_new = nullptr, *d_residual = nullptr;
int g_n = 0;
int g_nnz_r = 0;

// -----------------------------------------------------------------
// KERNEL V2: Uses separate R (off-diag) and D (diag) buffers
// -----------------------------------------------------------------
__global__ void jacobi_iteration_kernel_v2(
    const int* r_row_ptr,     // R matrix row pointers
    const int* r_col_idx,     // R matrix column indices
    const float* r_values,    // R matrix values (off-diagonal)
    const float* diag,        // D vector (a_ii values)
    const float* b,           // Right-hand side
    const float* x_old,       // Previous solution
    float* x_new,             // New solution (output)
    int n)                     // Matrix size
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        float sum = 0.0f; // This is the R*x_old part

        int row_start = r_row_ptr[i];
        int row_end = r_row_ptr[i + 1];

        // This loop now *only* calculates the sum over off-diagonals
        for (int j = row_start; j < row_end; ++j) {
            sum += r_values[j] * x_old[r_col_idx[j]];
        }

        // Get the diagonal value directly from the 'diag' array
        float diag_val = diag[i]; 

        // Jacobi update: x_new[i] = (b[i] - sum) / diag_val
        if (diag_val != 0.0f) {
            x_new[i] = (b[i] - sum) / diag_val;
        } else {
            x_new[i] = 0.0f; 
        }
    }
}

// -----------------------------------------------------------------
// KERNEL V2: Computes residual as (R+D)*x - b
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
// NEW: setup_gpu_solver
// Splits A -> R, D. Allocates GPU mem. Uploads constant data (R).
// -----------------------------------------------------------------
void setup_gpu_solver(const Eigen::SparseMatrix<float, Eigen::RowMajor>& A,
                      Eigen::VectorXf& D_host) 
{
    g_n = A.rows();
    std::vector<Eigen::Triplet<float>> r_triplets;
    D_host.resize(g_n);

    std::cout << "Splitting A into D (diagonal) and R (off-diagonal)...\n";
    for (int k = 0; k < A.outerSize(); ++k) {
        for (Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(A, k); it; ++it) {
            if (it.row() == it.col()) {
                D_host(it.row()) = it.value();
            } else {
                r_triplets.emplace_back(it.row(), it.col(), it.value());
            }
        }
    }

    Eigen::SparseMatrix<float, Eigen::RowMajor> R_host(g_n, g_n);
    R_host.setFromTriplets(r_triplets.begin(), r_triplets.end());
    R_host.makeCompressed();

    g_nnz_r = R_host.nonZeros();
    std::cout << "R matrix non-zeros: " << g_nnz_r << "\n\n";

    // --- Allocate ALL GPU memory ---
    CUDA_CHECK(cudaMalloc(&d_r_row_ptr, (g_n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_r_col_idx, g_nnz_r * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_r_values, g_nnz_r * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_diag, g_n * sizeof(float))); 
    CUDA_CHECK(cudaMalloc(&d_b, g_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x_old, g_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x_new, g_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_residual, g_n * sizeof(float)));

    // --- Perform ONE-TIME upload for unchanging R matrix ---
    std::cout << "Uploading R (off-diagonals) to GPU...\n";
    CUDA_CHECK(cudaMemcpy(d_r_row_ptr, R_host.outerIndexPtr(), (g_n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_r_col_idx, R_host.innerIndexPtr(), g_nnz_r * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_r_values, R_host.valuePtr(), g_nnz_r * sizeof(float), cudaMemcpyHostToDevice));
}

// -----------------------------------------------------------------
// NEW: cleanup_gpu_solver
// Frees all persistent GPU memory.
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
}

// -----------------------------------------------------------------
// MODIFIED: gpu_jacobi_solve
// Now assumes R is already on GPU. Only uploads D and b.
// -----------------------------------------------------------------
void gpu_jacobi_solve(const Eigen::VectorXf& D_host, // Pass host diagonal
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
    // This is the key optimization: only upload what changed!
    CUDA_CHECK(cudaMemcpy(d_diag, D_host.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b_host.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    // Initialize x = 0
    CUDA_CHECK(cudaMemset(d_x_old, 0, n * sizeof(float)));

    // Kernel launch parameters
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    std::cout << "Starting GPU Jacobi iterations...\n";
    std::cout << "Block size: " << block_size << ", Grid size: " << grid_size << "\n\n";

    // Allocate host memory for checking convergence
    std::vector<float> h_residual(n);
    bool converged = false;
    int final_iter = 0;

    // Jacobi iteration loop
    for (int iter = 0; iter < max_iter; ++iter) {
        final_iter = iter;
        // Perform one Jacobi iteration
        jacobi_iteration_kernel_v2<<<grid_size, block_size>>>(
            d_r_row_ptr, d_r_col_idx, d_r_values, 
            d_diag, // Pass the newly updated diagonal!
            d_b, 
            d_x_old, d_x_new, n
        );
        CUDA_CHECK(cudaGetLastError());

        // Check convergence every 10 iterations
        if (iter % 10 == 0 || iter == max_iter - 1) {
            // Compute residual: r = (R+D)*x - b
            // Note: we must check d_x_new, as it holds the latest result
            compute_residual_kernel_v2<<<grid_size, block_size>>>(
                d_r_row_ptr, d_r_col_idx, d_r_values, d_diag,
                d_x_new, // Check the solution we just computed
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

            std::cout << "Iteration " << iter << ": residual = " << residual_norm << "\n";

            if (residual_norm < tol) {
                std::cout << "\n✓ GPU converged in " << iter << " iterations!\n";
                converged = true;
                break;
            }
        }

        // Swap x_old and x_new for next iteration
        float* temp = d_x_old;
        d_x_old = d_x_new;
        d_x_new = temp;
    }
    
    if (!converged) {
        std::cout << "\nGPU reached max iterations (" << max_iter << ")\n";
    }

    // After the loop, the *final* valid solution is always in d_x_new
    // This is because we check convergence *after* the kernel writes to d_x_new
    // and *before* the swap.
    // If we break, d_x_new has the answer.
    // If the loop finishes, d_x_new has the answer from the last iteration.
    // ...UNLESS the last iteration was the 'swap'. Let's fix this.

    // A safer swap logic:
    // The solution we want to check is always in d_x_new.
    // If it converges, we copy d_x_new back.
    // If it doesn't converge, we swap, and the *next* iteration starts.
    // At the end, d_x_old points to the solution from iter N-1, and
    // d_x_new points to the solution from iter N.
    // Let's re-think the loop logic slightly to be cleaner.

    // --- Re-designed Loop Logic (Simpler) ---
    
    // We will always read from d_x_old and write to d_x_new
    // Then we swap *before* checking, so d_x_old always has the latest result.
    
    CUDA_CHECK(cudaMemset(d_x_old, 0, n * sizeof(float))); // Initial guess
    
    for (int iter = 0; iter < max_iter; ++iter) {
        // 1. Solve: x_new = f(x_old)
        jacobi_iteration_kernel_v2<<<grid_size, block_size>>>(
            d_r_row_ptr, d_r_col_idx, d_r_values, d_diag,
            d_b, d_x_old, d_x_new, n
        );
        CUDA_CHECK(cudaGetLastError());
        
        // 2. Swap: x_old is now the latest solution
        float* temp = d_x_old;
        d_x_old = d_x_new;
        d_x_new = temp;
        
        final_iter = iter + 1;

        // 3. Check: r = A*x_old - b
        if (iter % 10 == 0 || iter == max_iter - 1) {
            compute_residual_kernel_v2<<<grid_size, block_size>>>(
                d_r_row_ptr, d_r_col_idx, d_r_values, d_diag,
                d_x_old, // Check the solution we just swapped into d_x_old
                d_b, d_residual, n
            );
            // ... (copy residual, compute norm) ...
            CUDA_CHECK(cudaMemcpy(h_residual.data(), d_residual, n * sizeof(float), cudaMemcpyDeviceToHost));
            float residual_norm = 0.0f;
            for (int i = 0; i < n; ++i) residual_norm += h_residual[i] * h_residual[i];
            residual_norm = std::sqrt(residual_norm);

            std::cout << "Iteration " << iter+1 << ": residual = " << residual_norm << "\n";
            
            if (residual_norm < tol) {
                std::cout << "\n✓ GPU converged in " << final_iter << " iterations!\n";
                converged = true;
                break;
            }
        }
    }
    
    if (!converged) {
         std::cout << "\nGPU reached max iterations (" << max_iter << ")\n";
    }

    // --- Copy solution back to host ---
    // The final, correct solution is now always in d_x_old
    std::cout << "Transferring solution from GPU to CPU...\n";
    CUDA_CHECK(cudaMemcpy(x_host.data(), d_x_old, n * sizeof(float), cudaMemcpyDeviceToHost));
}


// -----------------------------------------------------------------
// (CPU SOLVER - Unchanged)
// -----------------------------------------------------------------
void cpu_jacobi_solve(const Eigen::SparseMatrix<float, Eigen::RowMajor>& A,
                      const Eigen::VectorXf& b,
                      Eigen::VectorXf& x,
                      int max_iter,
                      float tol) {
    int n = A.rows();
    Eigen::VectorXf x_new(n);
    x.setZero();

    for (int iter = 0; iter < max_iter; ++iter) {
        for (int i = 0; i < n; ++i) {
            float diag = 0.0f;
            float sum = 0.0f;
            for (Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(A, i); it; ++it) {
                if (it.col() == i) {
                    diag = it.value();
                } else {
                    sum += it.value() * x(it.col());
                }
            }
            if (diag != 0.0f) {
                x_new(i) = (b(i) - sum) / diag;
            } else {
                x_new(i) = 0.0f;
            }
        }

        float error = (x_new - x).norm();
        x = x_new; // Update x for next iteration

        if (iter % 10 == 0 || iter == max_iter - 1) {
            Eigen::VectorXf residual = A * x - b;
            float res_norm = residual.norm();
            std::cout << "Iteration " << iter+1 << ": residual = " << res_norm << "\n";
            if (res_norm < tol) {
                std::cout << "CPU converged in " << iter + 1 << " iterations\n";
                return;
            }
        }
    }
    std::cout << "CPU reached max iterations (" << max_iter << ")\n";
}


// Create a diagonally dominant SPD matrix
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
// (MAIN)
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

    // --- Large test ---
    std::cout << "========================================\n";
    std::cout << "Large Test (n=100000)\n";
    std::cout << "========================================\n";

    int n_large = 100000;
    Eigen::SparseMatrix<float, Eigen::RowMajor> A_large = create_diag_dominant_matrix(n_large);
    Eigen::VectorXf b_large = Eigen::VectorXf::Random(n_large);
    Eigen::VectorXf x_cpu_large(n_large);
    Eigen::VectorXf x_gpu_large(n_large);
    
    // This will hold the diagonal elements
    Eigen::VectorXf D_host_large; 

    std::cout << "Matrix size: " << n_large << "x" << n_large << "\n";
    std::cout << "Non-zeros: " << A_large.nonZeros() << "\n\n";

    // --- CPU solve ---
    std::cout << "--- Running CPU Jacobi ---\n";
    auto t_cpu_large_start = std::chrono::high_resolution_clock::now();
    cpu_jacobi_solve(A_large, b_large, x_cpu_large, max_iter, tol);
    auto t_cpu_large_end = std::chrono::high_resolution_clock::now();
    auto cpu_large_time = std::chrono::duration_cast<std::chrono::microseconds>(t_cpu_large_end - t_cpu_large_start).count();
    std::cout << "----------------------------\n";


    // --- GPU solve (Optimized) ---
    std::cout << "\n--- Running GPU Jacobi ---\n";
    // 1. One-time setup
    auto t_setup_start = std::chrono::high_resolution_clock::now();
    setup_gpu_solver(A_large, D_host_large);
    auto t_setup_end = std::chrono::high_resolution_clock::now();
    auto setup_time = std::chrono::duration_cast<std::chrono::microseconds>(t_setup_end - t_setup_start).count();
    std::cout << "GPU Setup Time: " << setup_time << " us\n";

    // In a real sim, you might modify D_host_large here based on collisions
    // e.g., D_host_large(123) += 1000.0f;
    // We will just use the original D for a fair comparison.

    // 2. Run the solver
    auto t_gpu_large_start = std::chrono::high_resolution_clock::now();
    gpu_jacobi_solve(D_host_large, b_large, x_gpu_large, max_iter, tol);
    auto t_gpu_large_end = std::chrono::high_resolution_clock::now();
    auto gpu_large_time = std::chrono::duration_cast<std::chrono::microseconds>(t_gpu_large_end - t_gpu_large_start).count();

    // 3. One-time cleanup
    cleanup_gpu_solver();
    std::cout << "---------------------------\n";


    // --- Performance Summary ---
    std::cout << "\n========================================\n";
    std::cout << "Performance Summary\n";
    std::cout << "========================================\n";
    std::cout << "CPU time: " << cpu_large_time << " μs (" 
              << (float)cpu_large_time / 1000.0f << " ms)\n";
    std::cout << "GPU time: " << gpu_large_time << " μs (" 
              << (float)gpu_large_time / 1000.0f << " ms)\n";
    std::cout << "GPU Setup time: " << setup_time << " μs (" 
              << (float)setup_time / 1000.0f << " ms)\n";
    std::cout << "Speedup (Solve only): " << (float)cpu_large_time / gpu_large_time << "x\n\n";

    // --- Verification & Displaying Solution ---
    std::cout << "========================================\n";
    std::cout << "Verification\n";
    std::cout << "========================================\n";
    
    // Set precision for printing vectors
    std::cout << std::fixed << std::setprecision(6);
    
    std::cout << "First 10 elements of x_cpu:\n[";
    for(int i = 0; i < 10; ++i) std::cout << x_cpu_large(i) << (i == 9 ? "" : ", ");
    std::cout << " ... ]\n";
    
    std::cout << "First 10 elements of x_gpu:\n[";
    for(int i = 0; i < 10; ++i) std::cout << x_gpu_large(i) << (i == 9 ? "" : ", ");
    std::cout << " ... ]\n\n";
    
    float error_large = (x_cpu_large - x_gpu_large).norm() / x_cpu_large.norm();
    Eigen::VectorXf residual_cpu_large = A_large * x_cpu_large - b_large;
    Eigen::VectorXf residual_gpu_large = A_large * x_gpu_large - b_large;

    std::cout << "Relative error ||x_cpu - x_gpu|| / ||x_cpu||: " << error_large << "\n";
    std::cout << "CPU residual ||A*x - b||: " << residual_cpu_large.norm() << "\n";
    std::cout << "GPU residual ||A*x - b||: " << residual_gpu_large.norm() << "\n";

    if (error_large < 1e-3f) {
        std::cout << "\n✓ Solutions match!\n";
    }

    return 0;
}