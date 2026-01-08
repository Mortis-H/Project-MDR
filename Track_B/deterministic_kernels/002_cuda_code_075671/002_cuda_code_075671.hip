// graph_coloring.cu
// A simple CUDA implementation for graph coloring using atomics
// Target architecture: Hopper

#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define NUM_NODES 4096
#define MAX_COLOR 16

// Kernel to initialize random colors
__global__ void initColors(int *colors, int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_NODES) {
        hiprandState state;
        hiprand_init(seed, idx, 0, &state);
        colors[idx] = hiprand(&state) % MAX_COLOR;
    }
}

// Kernel to perform graph coloring using atomics
__global__ void graphColoring(int *adjMatrix, int *colors) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_NODES) {
        int newColor = colors[idx];
        for (int j = 0; j < NUM_NODES; ++j) {
            if (adjMatrix[idx * NUM_NODES + j] == 1 && colors[j] == newColor) {
                // Atomically find a new color that is not used by neighbors
                int neighborColor = colors[j];
                int currentColor = newColor;
                while (currentColor == neighborColor) {
                    currentColor = atomicCAS(&colors[idx], currentColor, (currentColor + 1) % MAX_COLOR);
                }
                newColor = currentColor;
            }
        }
    }
}

int main() {
    int *h_colors, *d_colors, *h_adjMatrix, *d_adjMatrix;
    h_colors = (int *)malloc(NUM_NODES * sizeof(int));
    h_adjMatrix = (int *)malloc(NUM_NODES * NUM_NODES * sizeof(int));

    // Initialize adjacency matrix (simple ring graph for demonstration)
    for (int i = 0; i < NUM_NODES; ++i) {
        for (int j = 0; j < NUM_NODES; ++j) {
            h_adjMatrix[i * NUM_NODES + j] = (i == j - 1 || i == j + 1 || (i == 0 && j == NUM_NODES - 1) || (i == NUM_NODES - 1 && j == 0));
        }
    }

    // Allocate device memory
    hipError_t err = hipMalloc((void **)&d_colors, NUM_NODES * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Error allocating device memory for colors: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc((void **)&d_adjMatrix, NUM_NODES * NUM_NODES * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Error allocating device memory for adjacency matrix: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy adjacency matrix to device
    err = hipMemcpy(d_adjMatrix, h_adjMatrix, NUM_NODES * NUM_NODES * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Error copying adjacency matrix to device: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Initialize random colors on the device
    int blockSize = 256;
    int numBlocks = (NUM_NODES + blockSize - 1) / blockSize;
    initColors<<<numBlocks, blockSize>>>(d_colors, 1234);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error launching initColors kernel: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Perform graph coloring
    graphColoring<<<numBlocks, blockSize>>>(d_adjMatrix, d_colors);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error launching graphColoring kernel: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy results back to host
    err = hipMemcpy(h_colors, d_colors, NUM_NODES * sizeof(int), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Error copying colors back to host: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Print results
    std::cout << "Node colors after graph coloring:" << std::endl;
    for (int i = 0; i < NUM_NODES; ++i) {
        std::cout << "Node " << i << ": Color " << h_colors[i] << std::endl;
    }

    // Free device memory
    hipFree(d_colors);
    hipFree(d_adjMatrix);

    // Free host memory
    free(h_colors);
    free(h_adjMatrix);

    return 0;
}