#include <iostream>
#include <hip/hip_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>

// Kernel to compute the number of non-zero elements in the resulting matrix
__global__ void countNonZeroElements(int *nnzRow, int *rowPtrA, int *colIdxA, int *valA, int *rowPtrB, int *colIdxB, int *valB, int *nnzC, int numRowsA, int numColsA, int numColsB) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numRowsA) {
        int nnz = 0;
        for (int jj = rowPtrA[row]; jj < rowPtrA[row + 1]; ++jj) {
            int colA = colIdxA[jj];
            for (int kk = rowPtrB[colA]; kk < rowPtrB[colA + 1]; ++kk) {
                nnz++;
            }
        }
        nnzRow[row] = nnz;
    }
}

// Kernel to compute the row pointers for the resulting matrix
__global__ void computeRowPtr(int *rowPtrC, int *nnzRow, int numRowsA) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numRowsA) {
        rowPtrC[row + 1] = rowPtrC[row] + nnzRow[row];
    }
}

// Kernel to compute the column indices and values for the resulting matrix
__global__ void computeColIdxAndVal(int *rowPtrA, int *colIdxA, int *valA, int *rowPtrB, int *colIdxB, int *valB, int *rowPtrC, int *colIdxC, int *valC, int numRowsA, int numColsA, int numColsB) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numRowsA) {
        int startC = rowPtrC[row];
        for (int jj = rowPtrA[row]; jj < rowPtrA[row + 1]; ++jj) {
            int colA = colIdxA[jj];
            int valAij = valA[jj];
            for (int kk = rowPtrB[colA]; kk < rowPtrB[colA + 1]; ++kk) {
                int colB = colIdxB[kk];
                int valBjk = valB[kk];
                colIdxC[startC] = colB;
                valC[startC] = valAij * valBjk;
                startC++;
            }
        }
    }
}

// Main function to demonstrate the usage of the kernels
int main() {
    // Example sparse matrices in CSR format
    int numRowsA = 3, numColsA = 3, numColsB = 3;
    int rowPtrA[] = {0, 2, 3, 5};
    int colIdxA[] = {0, 2, 1, 0, 2};
    int valA[] = {1, 2, 3, 4, 5};

    int rowPtrB[] = {0, 3, 3, 4};
    int colIdxB[] = {0, 1, 2, 2};
    int valB[] = {6, 7, 8, 9};

    int *rowPtrC, *colIdxC, *valC;
    int *nnzRow, *nnzC;

    // Allocate Unified Memory
    hipMallocManaged(&rowPtrC, (numRowsA + 1) * sizeof(int));
    hipMallocManaged(&nnzRow, numRowsA * sizeof(int));
    hipMallocManaged(&nnzC, sizeof(int));
    hipMallocManaged(&colIdxC, numColsA * numColsB * sizeof(int));
    hipMallocManaged(&valC, numColsA * numColsB * sizeof(int));

    // Set up the grid and block dimensions
    int blockSize = 256;
    int numBlocks = (numRowsA + blockSize - 1) / blockSize;

    // Count non-zero elements in the resulting matrix
    countNonZeroElements<<<numBlocks, blockSize>>>(nnzRow, rowPtrA, colIdxA, valA, rowPtrB, colIdxB, valB, nnzC, numRowsA, numColsA, numColsB);
    hipDeviceSynchronize();

    // Compute the row pointers for the resulting matrix
    thrust::device_ptr<int> d_nnzRow(nnzRow);
    thrust::device_ptr<int> d_rowPtrC(rowPtrC);
    thrust::exclusive_scan(d_nnzRow, d_nnzRow + numRowsA, d_rowPtrC);
    hipDeviceSynchronize();

    // Compute the column indices and values for the resulting matrix
    computeColIdxAndVal<<<numBlocks, blockSize>>>(rowPtrA, colIdxA, valA, rowPtrB, colIdxB, valB, rowPtrC, colIdxC, valC, numRowsA, numColsA, numColsB);
    hipDeviceSynchronize();

    // Output the resulting matrix in CSR format
    std::cout << "Resulting Matrix C in CSR format:" << std::endl;
    std::cout << "rowPtrC: ";
    for (int i = 0; i <= numRowsA; ++i) {
        std::cout << rowPtrC[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "colIdxC: ";
    for (int i = 0; i < rowPtrC[numRowsA]; ++i) {
        std::cout << colIdxC[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "valC: ";
    for (int i = 0; i < rowPtrC[numRowsA]; ++i) {
        std::cout << valC[i] << " ";
    }
    std::cout << std::endl;

    // Free Unified Memory
    hipFree(rowPtrC);
    hipFree(nnzRow);
    hipFree(nnzC);
    hipFree(colIdxC);
    hipFree(valC);

    return 0;
}