#include <iostream>
#include <hip/hip_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>

// Define the number of nodes and edges
#define NUM_NODES 32
#define NUM_EDGES 48

// Adjacency list for the graph
__constant__ int adjList[NUM_EDGES * 2] = {
    0, 1, 0, 2, 0, 3, 1, 4, 1, 5, 2, 6, 2, 7, 3, 8, 3, 9, 4, 10, 4, 11, 5, 12, 5, 13,
    6, 14, 6, 15, 7, 16, 7, 17, 8, 18, 8, 19, 9, 20, 9, 21, 10, 22, 10, 23, 11, 24, 11, 25,
    12, 26, 12, 27, 13, 28, 13, 29, 14, 30, 14, 31, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
    25, 26, 27, 28, 29, 30, 31, 0
};

// Degree of each node
__constant__ int degrees[NUM_NODES] = {
    3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1
};

// Offset for each node in the adjacency list
__constant__ int offsets[NUM_NODES] = {
    0, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63
};

// Kernel to perform graph coloring
__global__ void graphColoringKernel(int* colors, int* maxColor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_NODES) return;

    int currentColor = 0;
    bool validColor = false;

    while (!validColor) {
        validColor = true;
        for (int i = offsets[idx]; i < offsets[idx] + degrees[idx]; ++i) {
            int neighbor = adjList[i];
            if (colors[neighbor] == currentColor) {
                validColor = false;
                break;
            }
        }
        if (!validColor) {
            currentColor++;
        }
    }

    colors[idx] = currentColor;
    atomicMax(maxColor, currentColor);
}

int main() {
    // Allocate memory for colors on the host
    thrust::host_vector<int> h_colors(NUM_NODES);
    thrust::fill(h_colors.begin(), h_colors.end(), -1); // Initialize colors to -1 (unassigned)

    // Allocate memory for colors on the device
    thrust::device_vector<int> d_colors = h_colors;

    // Allocate memory for max color on the device
    thrust::device_vector<int> d_maxColor(1);
    thrust::fill(d_maxColor.begin(), d_maxColor.end(), 0);

    // Launch the kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (NUM_NODES + threadsPerBlock - 1) / threadsPerBlock;
    graphColoringKernel<<<blocksPerGrid, threadsPerBlock>>>(thrust::raw_pointer_cast(d_colors.data()), thrust::raw_pointer_cast(d_maxColor.data()));

    // Check for errors in kernel launch
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy results back to host
    thrust::copy(d_colors.begin(), d_colors.end(), h_colors.begin());

    // Get the maximum color used
    int maxColor = d_maxColor[0] + 1;

    // Output the results
    std::cout << "Graph coloring result:" << std::endl;
    for (int i = 0; i < NUM_NODES; ++i) {
        std::cout << "Node " << i << ": Color " << h_colors[i] << std::endl;
    }
    std::cout << "Total number of colors used: " << maxColor << std::endl;

    return 0;
}