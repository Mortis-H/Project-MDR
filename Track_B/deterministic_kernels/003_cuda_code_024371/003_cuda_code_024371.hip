#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define NUM_NODES 2048
#define MAX_COLORS 16

__device__ int graph[NUM_NODES][NUM_NODES];

__global__ void initializeGraph(int *colors, hiprandState *states, int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_NODES) {
        hiprandState localState = states[idx];
        hiprand_init(seed, idx, 0, &localState);
        colors[idx] = hiprand(&localState) % MAX_COLORS;
        states[idx] = localState;
    }
}

__global__ void graphColoringKernel(int *colors, int *conflicts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_NODES) {
        int currentColor = colors[idx];
        conflicts[idx] = 0;
        for (int j = 0; j < NUM_NODES; j++) {
            if (graph[idx][j] && colors[j] == currentColor) {
                conflicts[idx] = 1;
                break;
            }
        }
    }
}

__global__ void resolveConflicts(int *colors, int *conflicts, hiprandState *states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_NODES && conflicts[idx]) {
        hiprandState localState = states[idx];
        int newColor = hiprand(&localState) % MAX_COLORS;
        while (newColor == colors[idx]) {
            newColor = hiprand(&localState) % MAX_COLORS;
        }
        colors[idx] = newColor;
        states[idx] = localState;
    }
}

void checkCudaError(hipError_t err, const char *msg) {
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << msg << " - " << hipGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    int *d_colors, *d_conflicts;
    hiprandState *d_states;
    int h_colors[NUM_NODES], h_conflicts[NUM_NODES];

    // Allocate memory on the device
    checkCudaError(hipMalloc(&d_colors, NUM_NODES * sizeof(int)), "hipMalloc d_colors");
    checkCudaError(hipMalloc(&d_conflicts, NUM_NODES * sizeof(int)), "hipMalloc d_conflicts");
    checkCudaError(hipMalloc(&d_states, NUM_NODES * sizeof(hiprandState)), "hipMalloc d_states");

    // Initialize random states on the device
    int blockSize = 256;
    int numBlocks = (NUM_NODES + blockSize - 1) / blockSize;
    initializeGraph<<<numBlocks, blockSize>>>(d_colors, d_states, 1234);
    checkCudaError(hipGetLastError(), "initializeGraph");

    // Simulate a simple graph (fully connected for simplicity)
    for (int i = 0; i < NUM_NODES; i++) {
        for (int j = 0; j < NUM_NODES; j++) {
            graph[i][j] = (i != j);
        }
    }

    // Graph coloring
    int maxIterations = 100;
    for (int iter = 0; iter < maxIterations; iter++) {
        graphColoringKernel<<<numBlocks, blockSize>>>(d_colors, d_conflicts);
        checkCudaError(hipGetLastError(), "graphColoringKernel");

        resolveConflicts<<<numBlocks, blockSize>>>(d_colors, d_conflicts, d_states);
        checkCudaError(hipGetLastError(), "resolveConflicts");

        // Check for convergence
        hipMemcpy(h_conflicts, d_conflicts, NUM_NODES * sizeof(int), hipMemcpyDeviceToHost);
        bool allResolved = true;
        for (int i = 0; i < NUM_NODES; i++) {
            if (h_conflicts[i]) {
                allResolved = false;
                break;
            }
        }
        if (allResolved) {
            std::cout << "Graph coloring converged in " << iter + 1 << " iterations." << std::endl;
            break;
        }
    }

    // Copy results back to host
    hipMemcpy(h_colors, d_colors, NUM_NODES * sizeof(int), hipMemcpyDeviceToHost);

    // Output the results
    std::cout << "Node colors:" << std::endl;
    for (int i = 0; i < NUM_NODES; i++) {
        std::cout << "Node " << i << ": Color " << h_colors[i] << std::endl;
    }

    // Free device memory
    hipFree(d_colors);
    hipFree(d_conflicts);
    hipFree(d_states);

    return 0;
}