#include <iostream>
#include <hip/hip_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/gather.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>

// Define the size of the matrix
#define N 64

// CUDA kernel to perform a simple relaxation step (e.g., Jacobi)
__global__ void jacobi_relaxation(const int* row_ptr, const int* col_ind, const float* val, const float* b, float* x_new, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sum = 0.0f;
        for (int i = row_ptr[idx]; i < row_ptr[idx + 1]; ++i) {
            if (col_ind[i] != idx) {
                sum += val[i] * x_new[col_ind[i]];
            }
        }
        x_new[idx] = (b[idx] - sum) / val[row_ptr[idx]];
    }
}

// CUDA kernel to perform a simple restriction step (e.g., full weighting)
__global__ void restriction(const int* row_ptr, const int* col_ind, const float* val, const float* r, float* r_coarse, int n, int n_coarse) {
    extern __shared__ float shared_r[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        shared_r[threadIdx.x] = r[idx];
        __syncthreads();
        if (threadIdx.x < n_coarse) {
            float sum = 0.0f;
            for (int i = row_ptr[threadIdx.x]; i < row_ptr[threadIdx.x + 1]; ++i) {
                sum += val[i] * shared_r[col_ind[i]];
            }
            atomicAdd(&r_coarse[threadIdx.x], sum);
        }
    }
}

// CUDA kernel to perform a simple prolongation step (e.g., bilinear interpolation)
__global__ void prolongation(const int* row_ptr, const int* col_ind, const float* val, const float* e_coarse, float* e, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sum = 0.0f;
        for (int i = row_ptr[idx]; i < row_ptr[idx + 1]; ++i) {
            sum += val[i] * e_coarse[col_ind[i]];
        }
        atomicAdd(&e[idx], sum);
    }
}

int main() {
    // Initialize matrix A and vector b (example data)
    thrust::host_vector<int> h_row_ptr(N + 1);
    thrust::host_vector<int> h_col_ind;
    thrust::host_vector<float> h_val;
    thrust::host_vector<float> h_b(N, 1.0f);
    thrust::host_vector<float> h_x(N, 0.0f);

    // Example matrix: diagonal dominance for simplicity
    int nnz = 0;
    for (int i = 0; i < N; ++i) {
        h_row_ptr[i] = nnz;
        if (i > 0) {
            h_col_ind.push_back(i - 1);
            h_val.push_back(-1.0f);
            nnz++;
        }
        h_col_ind.push_back(i);
        h_val.push_back(4.0f);
        nnz++;
        if (i < N - 1) {
            h_col_ind.push_back(i + 1);
            h_val.push_back(-1.0f);
            nnz++;
        }
    }
    h_row_ptr[N] = nnz;

    // Allocate device memory
    thrust::device_vector<int> d_row_ptr = h_row_ptr;
    thrust::device_vector<int> d_col_ind = h_col_ind;
    thrust::device_vector<float> d_val = h_val;
    thrust::device_vector<float> d_b = h_b;
    thrust::device_vector<float> d_x = h_x;

    // Multigrid parameters
    int n_coarse = N / 2;
    thrust::device_vector<float> d_r(N, 0.0f);
    thrust::device_vector<float> d_r_coarse(n_coarse, 0.0f);
    thrust::device_vector<float> d_e(n_coarse, 0.0f);
    thrust::device_vector<float> d_e_fine(N, 0.0f);

    // Set up CUDA execution parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Perform one V-cycle of multigrid
    // 1. Compute residual r = b - A*x
    thrust::fill(d_r.begin(), d_r.end(), 0.0f);
    jacobi_relaxation<<<blocksPerGrid, threadsPerBlock>>>(thrust::raw_pointer_cast(d_row_ptr.data()),
                                                        thrust::raw_pointer_cast(d_col_ind.data()),
                                                        thrust::raw_pointer_cast(d_val.data()),
                                                        thrust::raw_pointer_cast(d_b.data()),
                                                        thrust::raw_pointer_cast(d_r.data()),
                                                        N);
    hipDeviceSynchronize();
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error in Jacobi relaxation: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // 2. Restrict r to coarser grid
    thrust::fill(d_r_coarse.begin(), d_r_coarse.end(), 0.0f);
    restriction<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
        thrust::raw_pointer_cast(d_row_ptr.data()),
        thrust::raw_pointer_cast(d_col_ind.data()),
        thrust::raw_pointer_cast(d_val.data()),
        thrust::raw_pointer_cast(d_r.data()),
        thrust::raw_pointer_cast(d_r_coarse.data()),
        N, n_coarse);
    hipDeviceSynchronize();
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error in restriction: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // 3. Solve Ae = r_coarse on coarse grid (using Jacobi as a simple solver)
    thrust::fill(d_e.begin(), d_e.end(), 0.0f);
    jacobi_relaxation<<<blocksPerGrid / 2, threadsPerBlock>>>(thrust::raw_pointer_cast(d_row_ptr.data()),
                                                            thrust::raw_pointer_cast(d_col_ind.data()),
                                                            thrust::raw_pointer_cast(d_val.data()),
                                                            thrust::raw_pointer_cast(d_r_coarse.data()),
                                                            thrust::raw_pointer_cast(d_e.data()),
                                                            n_coarse);
    hipDeviceSynchronize();
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error in coarse grid solve: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // 4. Prolongate e to finer grid
    thrust::fill(d_e_fine.begin(), d_e_fine.end(), 0.0f);
    prolongation<<<blocksPerGrid, threadsPerBlock>>>(thrust::raw_pointer_cast(d_row_ptr.data()),
                                                   thrust::raw_pointer_cast(d_col_ind.data()),
                                                   thrust::raw_pointer_cast(d_val.data()),
                                                   thrust::raw_pointer_cast(d_e.data()),
                                                   thrust::raw_pointer_cast(d_e_fine.data()),
                                                   N);
    hipDeviceSynchronize();
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error in prolongation: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // 5. Correct solution x = x + e_fine
    thrust::transform(d_x.begin(), d_x.end(), d_e_fine.begin(), d_x.begin(), thrust::plus<float>());

    // Copy solution back to host and print
    thrust::host_vector<float> h_x_solution = d_x;
    std::cout << "Solution x:\n";
    for (int i = 0; i < N; ++i) {
        std::cout << h_x_solution[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}