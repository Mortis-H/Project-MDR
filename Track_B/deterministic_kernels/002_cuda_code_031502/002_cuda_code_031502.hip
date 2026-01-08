#include <iostream>
#include <hip/hip_runtime.h>


// Define the number of nodes in the graph
#define NUM_NODES 4096

// Kernel to initialize colors to -1 (uncolored)
__global__ void initializeColors(int* colors) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_NODES) {
        colors[idx] = -1;
    }
}

// Kernel to perform graph coloring
__global__ void colorGraph(int* adjMatrix, int* colors, int* numConflicts) {
    extern __shared__ int s_colors[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_NODES) return;

    // Load colors into shared memory
    s_colors[threadIdx.x] = colors[idx];
    __syncthreads();

    int color = 0;
    bool validColor = false;

    while (!validColor) {
        validColor = true;
        for (int j = 0; j < NUM_NODES; ++j) {
            if (adjMatrix[idx * NUM_NODES + j] && s_colors[j] == color) {
                validColor = false;
                break;
            }
        }
        if (!validColor) {
            color++;
        }
    }

    // Update color and check for conflicts
    if (atomicCAS(&colors[idx], -1, color) != -1) {
        atomicAdd(numConflicts, 1);
    }
}

int main() {
    // Allocate memory for adjacency matrix and colors on host
    int* h_adjMatrix = new int[NUM_NODES * NUM_NODES];
    int* h_colors = new int[NUM_NODES];
    int h_numConflicts = 0;

    // Initialize adjacency matrix (simple example: fully connected graph)
    for (int i = 0; i < NUM_NODES; ++i) {
        for (int j = 0; j < NUM_NODES; ++j) {
            h_adjMatrix[i * NUM_NODES + j] = (i != j);
        }
    }

    // Allocate memory for adjacency matrix and colors on device
    int* d_adjMatrix;
    int* d_colors;
    int* d_numConflicts;

    hipError_t err = hipMalloc(&d_adjMatrix, NUM_NODES * NUM_NODES * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for adjacency matrix" << std::endl;
        return -1;
    }

    err = hipMalloc(&d_colors, NUM_NODES * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for colors" << std::endl;
        hipFree(d_adjMatrix);
        return -1;
    }

    err = hipMalloc(&d_numConflicts, sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for numConflicts" << std::endl;
        hipFree(d_adjMatrix);
        hipFree(d_colors);
        return -1;
    }

    // Copy adjacency matrix to device
    err = hipMemcpy(d_adjMatrix, h_adjMatrix, NUM_NODES * NUM_NODES * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy adjacency matrix to device" << std::endl;
        hipFree(d_adjMatrix);
        hipFree(d_colors);
        hipFree(d_numConflicts);
        return -1;
    }

    // Initialize colors on device
    int blockSize = 256;
    int numBlocks = (NUM_NODES + blockSize - 1) / blockSize;
    initializeColors<<<numBlocks, blockSize>>>(d_colors);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch initializeColors kernel" << std::endl;
        hipFree(d_adjMatrix);
        hipFree(d_colors);
        hipFree(d_numConflicts);
        return -1;
    }

    // Initialize numConflicts to 0
    hipMemset(d_numConflicts, 0, sizeof(int));

    // Perform graph coloring
    do {
        h_numConflicts = 0;
        hipMemcpy(d_numConflicts, &h_numConflicts, sizeof(int), hipMemcpyHostToDevice);

        colorGraph<<<numBlocks, blockSize, blockSize * sizeof(int)>>>(d_adjMatrix, d_colors, d_numConflicts);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Failed to launch colorGraph kernel" << std::endl;
            hipFree(d_adjMatrix);
            hipFree(d_colors);
            hipFree(d_numConflicts);
            return -1;
        }

        hipMemcpy(&h_numConflicts, d_numConflicts, sizeof(int), hipMemcpyDeviceToHost);
    } while (h_numConflicts > 0);

    // Copy colors back to host
    err = hipMemcpy(h_colors, d_colors, NUM_NODES * sizeof(int), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy colors back to host" << std::endl;
        hipFree(d_adjMatrix);
        hipFree(d_colors);
        hipFree(d_numConflicts);
        return -1;
    }

    // Free device memory
    hipFree(d_adjMatrix);
    hipFree(d_colors);
    hipFree(d_numConflicts);

    // Output the colors
    std::cout << "Node colors:" << std::endl;
    for (int i = 0; i < NUM_NODES; ++i) {
        std::cout << "Node " << i << ": Color " << h_colors[i] << std::endl;
    }

    // Free host memory
    delete[] h_adjMatrix;
    delete[] h_colors;

    return 0;
}