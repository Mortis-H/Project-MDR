#include <iostream>
#include <hip/hip_runtime.h>
#include <vector>

// Kernel for sparse matrix-vector multiplication
__global__ void sparseMatrixVectorMul(const int *rowPtr, const int *colIdx, const float *values, const float *x, float *y, int numRows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numRows) {
        float sum = 0.0f;
        int rowStart = rowPtr[row];
        int rowEnd = rowPtr[row + 1];
        for (int j = rowStart; j < rowEnd; ++j) {
            sum += values[j] * x[colIdx[j]];
        }
        y[row] = sum;
    }
}

int main() {
    // Example matrix in CSR format
    // Matrix: 
    // [1 0 0 0 2]
    // [0 3 0 4 0]
    // [5 0 6 0 0]
    // [0 0 0 7 8]
    // [9 0 0 0 10]
    int numRows = 5;
    int nnz = 10; // Number of non-zero elements

    // CSR format data
    int h_rowPtr[] = {0, 2, 4, 6, 8, 10};
    int h_colIdx[] = {0, 4, 1, 3, 0, 2, 3, 4, 0, 4};
    float h_values[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    float h_x[] = {1, 2, 3, 4, 5}; // Input vector
    float h_y[numRows]; // Output vector

    // Allocate device memory
    int *d_rowPtr, *d_colIdx;
    float *d_values, *d_x, *d_y;
    hipMalloc((void **)&d_rowPtr, (numRows + 1) * sizeof(int));
    hipMalloc((void **)&d_colIdx, nnz * sizeof(int));
    hipMalloc((void **)&d_values, nnz * sizeof(float));
    hipMalloc((void **)&d_x, numRows * sizeof(float));
    hipMalloc((void **)&d_y, numRows * sizeof(float));

    // Copy data to device
    hipMemcpyAsync(d_rowPtr, h_rowPtr, (numRows + 1) * sizeof(int), hipMemcpyHostToDevice, 0);
    hipMemcpyAsync(d_colIdx, h_colIdx, nnz * sizeof(int), hipMemcpyHostToDevice, 0);
    hipMemcpyAsync(d_values, h_values, nnz * sizeof(float), hipMemcpyHostToDevice, 0);
    hipMemcpyAsync(d_x, h_x, numRows * sizeof(float), hipMemcpyHostToDevice, 0);

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (numRows + blockSize - 1) / blockSize;
    sparseMatrixVectorMul<<<numBlocks, blockSize>>>(d_rowPtr, d_colIdx, d_values, d_x, d_y, numRows);
    hipDeviceSynchronize(); // Ensure kernel execution completes

    // Copy result back to host
    hipMemcpy(h_y, d_y, numRows * sizeof(float), hipMemcpyDeviceToHost);

    // Print result
    std::cout << "Result vector y: ";
    for (int i = 0; i < numRows; ++i) {
        std::cout << h_y[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    hipFree(d_rowPtr);
    hipFree(d_colIdx);
    hipFree(d_values);
    hipFree(d_x);
    hipFree(d_y);

    return 0;
}