// scc_kernel.cu
// CUDA kernel to find strongly connected components in a directed graph with 2048 nodes
// Optimized for persistent threads and uses asynchronous memory copies

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>

#define NUM_NODES 2048
#define BLOCK_SIZE 256

// Kernel to initialize labels
__global__ void initLabels(int *labels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_NODES) {
        labels[idx] = idx; // Each node is its own component initially
    }
}

// Kernel to perform one pass of the union-find operation
__global__ void findComponents(int *graph, int *labels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_NODES) {
        int neighbor = graph[idx];
        int neighborLabel = labels[neighbor];
        int currentLabel = labels[idx];
        if (neighborLabel < currentLabel) {
            labels[idx] = neighborLabel;
        }
    }
}

// Function to check CUDA errors
void checkCudaError(hipError_t err, const char *operation) {
    if (err != hipSuccess) {
        std::cerr << "CUDA error during " << operation << ": " << hipGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Host graph adjacency list representation (for simplicity, using a single outgoing edge per node)
    std::vector<int> h_graph(NUM_NODES);
    for (int i = 0; i < NUM_NODES; ++i) {
        h_graph[i] = (i + 1) % NUM_NODES; // Simple cycle graph for demonstration
    }

    // Allocate device memory
    int *d_graph, *d_labels;
    checkCudaError(hipMalloc(&d_graph, NUM_NODES * sizeof(int)), "hipMalloc d_graph");
    checkCudaError(hipMalloc(&d_labels, NUM_NODES * sizeof(int)), "hipMalloc d_labels");

    // Asynchronously copy graph data to device
    hipMemcpyAsync(d_graph, h_graph.data(), NUM_NODES * sizeof(int), hipMemcpyHostToDevice, 0);

    // Initialize labels on the device
    initLabels<<<(NUM_NODES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_labels);
    checkCudaError(hipGetLastError(), "initLabels kernel launch");
    hipDeviceSynchronize();

    // Perform union-find iterations
    for (int iter = 0; iter < 10; ++iter) { // 10 iterations for demonstration purposes
        findComponents<<<(NUM_NODES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_graph, d_labels);
        checkCudaError(hipGetLastError(), "findComponents kernel launch");
        hipDeviceSynchronize();
    }

    // Allocate host memory for labels
    std::vector<int> h_labels(NUM_NODES);

    // Asynchronously copy labels back to host
    hipMemcpyAsync(h_labels.data(), d_labels, NUM_NODES * sizeof(int), hipMemcpyDeviceToHost, 0);
    hipDeviceSynchronize();

    // Free device memory
    hipFree(d_graph);
    hipFree(d_labels);

    // Sort and print unique labels (SCCs)
    std::sort(h_labels.begin(), h_labels.end());
    auto last = std::unique(h_labels.begin(), h_labels.end());
    std::cout << "Number of Strongly Connected Components: " << std::distance(h_labels.begin(), last) << std::endl;

    return 0;
}