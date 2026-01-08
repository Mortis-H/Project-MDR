#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define NUM_ELEMENTS 512
#define NUM_LEVELS 5
#define NUM_GPUS 2

// Node structure for the skip list
struct Node {
    int key;
    Node* next[NUM_LEVELS];
};

// Kernel to initialize nodes
__global__ void initializeNodes(Node* nodes, int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        nodes[idx].key = idx;
        for (int level = 0; level < NUM_LEVELS; ++level) {
            nodes[idx].next[level] = nullptr;
        }
    }
}

// Kernel to build the skip list
__global__ void buildSkipList(Node* nodes, int numElements, int numLevels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        for (int level = 0; level < numLevels; ++level) {
            if (idx + (1 << level) < numElements) {
                nodes[idx].next[level] = &nodes[idx + (1 << level)];
            }
        }
    }
}

// Function to check for CUDA errors
void checkCudaError(hipError_t err, const char* msg) {
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << msg << " (" << hipGetErrorString(err) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    Node* d_nodes[NUM_GPUS];
    int elementsPerGPU = NUM_ELEMENTS / NUM_GPUS;
    int threadsPerBlock = 256;
    int blocksPerGrid = (elementsPerGPU + threadsPerBlock - 1) / threadsPerBlock;

    // Initialize random number generator
    hiprandState* d_states[NUM_GPUS];
    for (int gpu = 0; gpu < NUM_GPUS; ++gpu) {
        hipSetDevice(gpu);
        hipMalloc(&d_nodes[gpu], elementsPerGPU * sizeof(Node));
        hipMalloc(&d_states[gpu], elementsPerGPU * sizeof(hiprandState));

        // Initialize nodes
        initializeNodes<<<blocksPerGrid, threadsPerBlock>>>(d_nodes[gpu], elementsPerGPU);
        checkCudaError(hipGetLastError(), "initializeNodes kernel launch failed");
        hipDeviceSynchronize();

        // Build skip list
        buildSkipList<<<blocksPerGrid, threadsPerBlock>>>(d_nodes[gpu], elementsPerGPU, NUM_LEVELS);
        checkCudaError(hipGetLastError(), "buildSkipList kernel launch failed");
        hipDeviceSynchronize();
    }

    // Free allocated memory
    for (int gpu = 0; gpu < NUM_GPUS; ++gpu) {
        hipSetDevice(gpu);
        hipFree(d_nodes[gpu]);
        hipFree(d_states[gpu]);
    }

    std::cout << "Skip list built successfully on " << NUM_GPUS << " GPUs." << std::endl;
    return 0;
}