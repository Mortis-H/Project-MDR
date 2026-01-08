#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define NUM_NODES 1024
#define MAX_DEGREE 16

__global__ void initializeGraph(int *degrees, int *edges, int *colors) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_NODES) {
        degrees[idx] = (idx % MAX_DEGREE) + 1; // Random degree between 1 and MAX_DEGREE
        colors[idx] = -1; // Initialize colors to -1 (unassigned)
        for (int i = 0; i < degrees[idx]; i++) {
            edges[idx * MAX_DEGREE + i] = (idx + i + 1) % NUM_NODES; // Simple circular graph
        }
    }
}

__global__ void graphColoring(int *degrees, int *edges, int *colors) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_NODES) {
        bool validColor;
        for (int color = 0; color < NUM_NODES; color++) {
            validColor = true;
            for (int i = 0; i < degrees[idx]; i++) {
                int neighbor = edges[idx * MAX_DEGREE + i];
                if (colors[neighbor] == color) {
                    validColor = false;
                    break;
                }
            }
            if (validColor) {
                colors[idx] = color;
                break;
            }
        }
    }
}

void checkCudaError(hipError_t err, const char *message) {
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << message << " - " << hipGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    int *h_degrees, *h_edges, *h_colors;
    int *d_degrees, *d_edges, *d_colors;

    // Allocate host memory
    h_degrees = (int *)malloc(NUM_NODES * sizeof(int));
    h_edges = (int *)malloc(NUM_NODES * MAX_DEGREE * sizeof(int));
    h_colors = (int *)malloc(NUM_NODES * sizeof(int));

    // Allocate device memory
    checkCudaError(hipMalloc((void **)&d_degrees, NUM_NODES * sizeof(int)), "Failed to allocate d_degrees");
    checkCudaError(hipMalloc((void **)&d_edges, NUM_NODES * MAX_DEGREE * sizeof(int)), "Failed to allocate d_edges");
    checkCudaError(hipMalloc((void **)&d_colors, NUM_NODES * sizeof(int)), "Failed to allocate d_colors");

    // Initialize device memory
    int blockSize = 256;
    int numBlocks = (NUM_NODES + blockSize - 1) / blockSize;
    initializeGraph<<<numBlocks, blockSize>>>(d_degrees, d_edges, d_colors);
    checkCudaError(hipGetLastError(), "Kernel launch failed");
    checkCudaError(hipDeviceSynchronize(), "Failed to synchronize");

    // Perform graph coloring
    graphColoring<<<numBlocks, blockSize>>>(d_degrees, d_edges, d_colors);
    checkCudaError(hipGetLastError(), "Kernel launch failed");
    checkCudaError(hipDeviceSynchronize(), "Failed to synchronize");

    // Copy results back to host
    checkCudaError(hipMemcpy(h_colors, d_colors, NUM_NODES * sizeof(int), hipMemcpyDeviceToHost), "Failed to copy colors to host");

    // Print results
    for (int i = 0; i < 10; i++) { // Print first 10 nodes for brevity
        std::cout << "Node " << i << " has color " << h_colors[i] << std::endl;
    }

    // Free device memory
    hipFree(d_degrees);
    hipFree(d_edges);
    hipFree(d_colors);

    // Free host memory
    free(h_degrees);
    free(h_edges);
    free(h_colors);

    return 0;
}