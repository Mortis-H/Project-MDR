#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

namespace cg = cooperative_groups;

// Define the number of non-zero elements and matrix size
#define NNZ 64
#define N 1024

// Kernel for sparse matrix-vector multiplication using CSR format
__global__ void csrSpMV(int *rowPtr, int *colInd, float *values, float *x, float *y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N) {
        float dot = 0.0f;
        int row_start = rowPtr[row];
        int row_end = rowPtr[row + 1];
        for (int jj = row_start; jj < row_end; jj++) {
            int col = colInd[jj];
            float val = values[jj];
            dot += val * x[col];
        }
        y[row] = dot;
    }
}

// Kernel using Cooperative Groups for better synchronization
__global__ void csrSpMVCoopGroups(int *rowPtr, int *colInd, float *values, float *x, float *y) {
    cg::thread_block cta = cg::this_thread_block();
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    float dot = 0.0f;

    if (row < N) {
        int row_start = rowPtr[row];
        int row_end = rowPtr[row + 1];
        for (int jj = row_start; jj < row_end; jj++) {
            int col = colInd[jj];
            float val = values[jj];
            dot += val * x[col];
        }
    }

    // Use cooperative groups to ensure all threads in the block have completed their work
    cg::sync(cta);
    if (row < N) {
        y[row] = dot;
    }
}

int main() {
    // Host data
    int h_rowPtr[N + 1];
    int h_colInd[NNZ];
    float h_values[NNZ];
    float h_x[N];
    float h_y[N];

    // Initialize matrix in CSR format and vector x
    // This is a simplified initialization for demonstration purposes
    for (int i = 0; i < N; ++i) {
        h_rowPtr[i] = i * (NNZ / N);
        h_x[i] = static_cast<float>(i + 1);
        h_y[i] = 0.0f;
    }
    h_rowPtr[N] = NNZ;
    for (int i = 0; i < NNZ; ++i) {
        h_colInd[i] = i % N;
        h_values[i] = static_cast<float>(i + 1);
    }

    // Device data
    int *d_rowPtr, *d_colInd;
    float *d_values, *d_x, *d_y;

    // Allocate device memory
    hipError_t err = hipMalloc((void**)&d_rowPtr, (N + 1) * sizeof(int));
    if (err != hipSuccess) { std::cerr << "Error allocating d_rowPtr: " << hipGetErrorString(err) << std::endl; return -1; }
    err = hipMalloc((void**)&d_colInd, NNZ * sizeof(int));
    if (err != hipSuccess) { std::cerr << "Error allocating d_colInd: " << hipGetErrorString(err) << std::endl; return -1; }
    err = hipMalloc((void**)&d_values, NNZ * sizeof(float));
    if (err != hipSuccess) { std::cerr << "Error allocating d_values: " << hipGetErrorString(err) << std::endl; return -1; }
    err = hipMalloc((void**)&d_x, N * sizeof(float));
    if (err != hipSuccess) { std::cerr << "Error allocating d_x: " << hipGetErrorString(err) << std::endl; return -1; }
    err = hipMalloc((void**)&d_y, N * sizeof(float));
    if (err != hipSuccess) { std::cerr << "Error allocating d_y: " << hipGetErrorString(err) << std::endl; return -1; }

    // Copy data from host to device
    err = hipMemcpy(d_rowPtr, h_rowPtr, (N + 1) * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) { std::cerr << "Error copying d_rowPtr: " << hipGetErrorString(err) << std::endl; return -1; }
    err = hipMemcpy(d_colInd, h_colInd, NNZ * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) { std::cerr << "Error copying d_colInd: " << hipGetErrorString(err) << std::endl; return -1; }
    err = hipMemcpy(d_values, h_values, NNZ * sizeof(float), hipMemcpyHostToDevice);
    if (err != hipSuccess) { std::cerr << "Error copying d_values: " << hipGetErrorString(err) << std::endl; return -1; }
    err = hipMemcpy(d_x, h_x, N * sizeof(float), hipMemcpyHostToDevice);
    if (err != hipSuccess) { std::cerr << "Error copying d_x: " << hipGetErrorString(err) << std::endl; return -1; }

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    csrSpMVCoopGroups<<<numBlocks, blockSize>>>(d_rowPtr, d_colInd, d_values, d_x, d_y);
    err = hipGetLastError();
    if (err != hipSuccess) { std::cerr << "Error launching kernel: " << hipGetErrorString(err) << std::endl; return -1; }

    // Copy result back to host
    err = hipMemcpy(h_y, d_y, N * sizeof(float), hipMemcpyDeviceToHost);
    if (err != hipSuccess) { std::cerr << "Error copying d_y: " << hipGetErrorString(err) << std::endl; return -1; }

    // Free device memory
    hipFree(d_rowPtr);
    hipFree(d_colInd);
    hipFree(d_values);
    hipFree(d_x);
    hipFree(d_y);

    // Output the first 10 elements of the result vector y for verification
    std::cout << "Result vector y (first 10 elements): ";
    for (int i = 0; i < 10; ++i) {
        std::cout << h_y[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}