#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define N 16384
#define BLOCK_SIZE 256

// Kernel to initialize the grid with random values
__global__ void initializeGrid(float *grid, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        hiprandState state;
        hiprand_init((unsigned long long) clock() + idx, 0, 0, &state);
        grid[idx] = hiprand_uniform(&state) * 100.0f; // Random value between 0 and 100
    }
}

// Kernel to perform the heat equation update
__global__ void heatEquationKernel(float *grid, float *newGrid, int n, float alpha) {
    __shared__ float sharedGrid[BLOCK_SIZE + 2]; // Extra space for halo cells

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory with halo cells
    if (idx < n) {
        if (threadIdx.x == 0 && blockIdx.x > 0) {
            sharedGrid[0] = grid[idx - blockDim.x]; // Load left halo
        }
        if (threadIdx.x == blockDim.x - 1 && blockIdx.x < (n + blockDim.x - 1) / blockDim.x - 1) {
            sharedGrid[BLOCK_SIZE + 1] = grid[idx + blockDim.x]; // Load right halo
        }
        sharedGrid[threadIdx.x + 1] = grid[idx];
    }

    __syncthreads(); // Ensure all data is loaded into shared memory

    // Update the grid using the heat equation
    if (idx > 0 && idx < n - 1) {
        newGrid[idx] = grid[idx] + alpha * (sharedGrid[threadIdx.x] - 2.0f * sharedGrid[threadIdx.x + 1] + sharedGrid[threadIdx.x + 2]);
    } else if (idx == 0) {
        newGrid[idx] = grid[idx] + alpha * (0.0f - 2.0f * sharedGrid[threadIdx.x + 1] + sharedGrid[threadIdx.x + 2]);
    } else if (idx == n - 1) {
        newGrid[idx] = grid[idx] + alpha * (sharedGrid[threadIdx.x] - 2.0f * sharedGrid[threadIdx.x + 1] + 0.0f);
    }
}

int main() {
    float *d_grid, *d_newGrid;
    float *h_grid = new float[N];
    float alpha = 0.01f; // Diffusion coefficient

    // Allocate memory on the device
    hipError_t err = hipMalloc((void**)&d_grid, N * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device grid: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc((void**)&d_newGrid, N * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device newGrid: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Initialize the grid with random values
    initializeGrid<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_grid, N);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch initializeGrid kernel: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Perform the heat equation update
    heatEquationKernel<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_grid, d_newGrid, N, alpha);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch heatEquationKernel kernel: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy the result back to the host
    err = hipMemcpy(h_grid, d_newGrid, N * sizeof(float), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy result from device to host: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Free device memory
    hipFree(d_grid);
    hipFree(d_newGrid);

    // Clean up
    delete[] h_grid;

    return 0;
}