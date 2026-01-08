#include <iostream>
#include <hip/hip_runtime.h>


#define NUM_ELEMENTS 2048
#define NUM_GPUS 2

// Kernel to initialize binary tree nodes
__global__ void initializeTree(int* tree, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        tree[idx] = idx; // Simple initialization, can be replaced with any logic
    }
}

// Kernel to traverse binary tree (in-order traversal example)
__global__ void traverseTree(int* tree, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // In-order traversal logic: left, root, right
        // This is a simplified example, actual traversal logic depends on tree structure
        if (2 * idx + 1 < n) {
            // Left child
            tree[idx] += tree[2 * idx + 1];
        }
        if (2 * idx + 2 < n) {
            // Right child
            tree[idx] += tree[2 * idx + 2];
        }
    }
}

// Function to check CUDA errors
void checkCudaError(hipError_t err, const char* msg) {
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << msg << " (" << hipGetErrorString(err) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    int h_tree[NUM_ELEMENTS];
    int* d_trees[NUM_GPUS];
    int elementsPerGPU = NUM_ELEMENTS / NUM_GPUS;
    int threadsPerBlock = 256;
    int blocksPerGrid = (elementsPerGPU + threadsPerBlock - 1) / threadsPerBlock;

    // Initialize each GPU
    for (int gpu = 0; gpu < NUM_GPUS; ++gpu) {
        hipSetDevice(gpu);
        hipMalloc(&d_trees[gpu], elementsPerGPU * sizeof(int));
        checkCudaError(hipGetLastError(), "hipMalloc failed");
    }

    // Launch initialization kernels on each GPU
    for (int gpu = 0; gpu < NUM_GPUS; ++gpu) {
        hipSetDevice(gpu);
        initializeTree<<<blocksPerGrid, threadsPerBlock>>>(d_trees[gpu], elementsPerGPU);
        checkCudaError(hipGetLastError(), "initializeTree kernel launch failed");
        hipDeviceSynchronize();
        checkCudaError(hipGetLastError(), "hipDeviceSynchronize failed");
    }

    // Launch traversal kernels on each GPU
    for (int gpu = 0; gpu < NUM_GPUS; ++gpu) {
        hipSetDevice(gpu);
        traverseTree<<<blocksPerGrid, threadsPerBlock>>>(d_trees[gpu], elementsPerGPU);
        checkCudaError(hipGetLastError(), "traverseTree kernel launch failed");
        hipDeviceSynchronize();
        checkCudaError(hipGetLastError(), "hipDeviceSynchronize failed");
    }

    // Copy results back to host for verification
    for (int gpu = 0; gpu < NUM_GPUS; ++gpu) {
        hipSetDevice(gpu);
        hipMemcpy(&h_tree[gpu * elementsPerGPU], d_trees[gpu], elementsPerGPU * sizeof(int), hipMemcpyDeviceToHost);
        checkCudaError(hipGetLastError(), "hipMemcpy failed");
    }

    // Free device memory
    for (int gpu = 0; gpu < NUM_GPUS; ++gpu) {
        hipSetDevice(gpu);
        hipFree(d_trees[gpu]);
        checkCudaError(hipGetLastError(), "hipFree failed");
    }

    // Output the result for verification (first 10 elements)
    std::cout << "First 10 elements of the processed tree:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << h_tree[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}