#include <iostream>
#include <hip/hip_runtime.h>


// Define the number of nodes and edges for the graph
#define NUM_NODES 512
#define NUM_EDGES 1024

// Kernel to initialize the colors of each node to -1 (unassigned)
__global__ void initializeColors(int *colors) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_NODES) {
        colors[idx] = -1;
    }
}

// Kernel to perform graph coloring
__global__ void graphColoring(int *adjMatrix, int *colors, int maxColors) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_NODES) {
        int color = 0;
        bool validColor = false;
        while (!validColor && color < maxColors) {
            validColor = true;
            for (int j = 0; j < NUM_NODES; j++) {
                if (adjMatrix[idx * NUM_NODES + j] && colors[j] == color) {
                    validColor = false;
                    break;
                }
            }
            if (!validColor) {
                color++;
            }
        }
        colors[idx] = color;
    }
}

// Function to check for CUDA errors
void checkCudaError(hipError_t err, const char *function, int line) {
    if (err != hipSuccess) {
        std::cerr << "CUDA error at " << function << ":" << line << " - " << hipGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Allocate memory for the adjacency matrix and colors on the host
    int *h_adjMatrix = new int[NUM_NODES * NUM_NODES];
    int *h_colors = new int[NUM_NODES];

    // Initialize the adjacency matrix with random values (0 or 1)
    // For simplicity, we'll create a random graph here
    for (int i = 0; i < NUM_NODES; i++) {
        for (int j = 0; j < NUM_NODES; j++) {
            h_adjMatrix[i * NUM_NODES + j] = (i != j) ? (rand() % 2) : 0;
        }
    }

    // Allocate memory for the adjacency matrix and colors on the device
    int *d_adjMatrix, *d_colors;
    checkCudaError(hipMalloc((void**)&d_adjMatrix, NUM_NODES * NUM_NODES * sizeof(int)), __FUNCTION__, __LINE__);
    checkCudaError(hipMalloc((void**)&d_colors, NUM_NODES * sizeof(int)), __FUNCTION__, __LINE__);

    // Copy the adjacency matrix from the host to the device
    checkCudaError(hipMemcpy(d_adjMatrix, h_adjMatrix, NUM_NODES * NUM_NODES * sizeof(int), hipMemcpyHostToDevice), __FUNCTION__, __LINE__);

    // Initialize the colors array on the device
    int blockSize = 256;
    int numBlocks = (NUM_NODES + blockSize - 1) / blockSize;
    initializeColors<<<numBlocks, blockSize>>>(d_colors);
    checkCudaError(hipGetLastError(), __FUNCTION__, __LINE__);
    checkCudaError(hipDeviceSynchronize(), __FUNCTION__, __LINE__);

    // Perform graph coloring
    graphColoring<<<numBlocks, blockSize>>>(d_adjMatrix, d_colors, NUM_NODES);
    checkCudaError(hipGetLastError(), __FUNCTION__, __LINE__);
    checkCudaError(hipDeviceSynchronize(), __FUNCTION__, __LINE__);

    // Copy the colors array from the device to the host
    checkCudaError(hipMemcpy(h_colors, d_colors, NUM_NODES * sizeof(int), hipMemcpyDeviceToHost), __FUNCTION__, __LINE__);

    // Print the colors assigned to each node
    std::cout << "Colors assigned to each node:" << std::endl;
    for (int i = 0; i < NUM_NODES; i++) {
        std::cout << "Node " << i << ": Color " << h_colors[i] << std::endl;
    }

    // Free the device memory
    checkCudaError(hipFree(d_adjMatrix), __FUNCTION__, __LINE__);
    checkCudaError(hipFree(d_colors), __FUNCTION__, __LINE__);

    // Free the host memory
    delete[] h_adjMatrix;
    delete[] h_colors;

    return 0;
}