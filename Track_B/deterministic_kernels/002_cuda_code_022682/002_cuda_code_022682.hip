// scc_cuda.cu
// CUDA kernel for finding strongly connected components in a directed graph with 512 nodes
// Optimized for compute throughput on NVIDIA Hopper architecture using CUDA Unified Memory

#include <iostream>
#include <hip/hip_runtime.h>

#define NUM_NODES 512
#define MAX_EDGES 2048

__global__ void propagateLabels(int* labels, int* adjacencyMatrix, int* changed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_NODES) return;

    int originalLabel = labels[idx];
    for (int j = 0; j < NUM_NODES; ++j) {
        if (adjacencyMatrix[idx * NUM_NODES + j] == 1 && labels[j] < labels[idx]) {
            labels[idx] = labels[j];
            *changed = 1;
        }
    }
    if (originalLabel != labels[idx]) {
        *changed = 1;
    }
}

__global__ void compressLabels(int* labels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_NODES) return;

    labels[idx] = labels[labels[idx]];
}

int main() {
    // Initialize CUDA
    hipError_t err;

    // Allocate Unified Memory for labels and adjacency matrix
    int* labels;
    int* adjacencyMatrix;
    int* changed;

    err = hipMallocManaged(&labels, NUM_NODES * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate managed memory for labels" << std::endl;
        return -1;
    }

    err = hipMallocManaged(&adjacencyMatrix, NUM_NODES * NUM_NODES * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate managed memory for adjacency matrix" << std::endl;
        hipFree(labels);
        return -1;
    }

    err = hipMallocManaged(&changed, sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate managed memory for changed flag" << std::endl;
        hipFree(labels);
        hipFree(adjacencyMatrix);
        return -1;
    }

    // Initialize labels and adjacency matrix
    for (int i = 0; i < NUM_NODES; ++i) {
        labels[i] = i;
    }

    // Example adjacency matrix initialization (random graph with MAX_EDGES edges)
    for (int i = 0; i < NUM_NODES * NUM_NODES; ++i) {
        adjacencyMatrix[i] = 0;
    }
    for (int i = 0; i < MAX_EDGES; ++i) {
        int src = rand() % NUM_NODES;
        int dst = rand() % NUM_NODES;
        adjacencyMatrix[src * NUM_NODES + dst] = 1;
    }

    // Kernel launch configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (NUM_NODES + threadsPerBlock - 1) / threadsPerBlock;

    // Perform the path compression until no more changes are made
    do {
        *changed = 0;
        propagateLabels<<<blocksPerGrid, threadsPerBlock>>>(labels, adjacencyMatrix, changed);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Kernel propagateLabels launch failed: " << hipGetErrorString(err) << std::endl;
            hipFree(labels);
            hipFree(adjacencyMatrix);
            hipFree(changed);
            return -1;
        }

        compressLabels<<<blocksPerGrid, threadsPerBlock>>>(labels);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Kernel compressLabels launch failed: " << hipGetErrorString(err) << std::endl;
            hipFree(labels);
            hipFree(adjacencyMatrix);
            hipFree(changed);
            return -1;
        }

        err = hipDeviceSynchronize();
        if (err != hipSuccess) {
            std::cerr << "Failed to synchronize: " << hipGetErrorString(err) << std::endl;
            hipFree(labels);
            hipFree(adjacencyMatrix);
            hipFree(changed);
            return -1;
        }
    } while (*changed);

    // Print the labels (SCCs)
    std::cout << "Strongly Connected Components:" << std::endl;
    for (int i = 0; i < NUM_NODES; ++i) {
        std::cout << "Node " << i << " -> SCC " << labels[i] << std::endl;
    }

    // Free Unified Memory
    hipFree(labels);
    hipFree(adjacencyMatrix);
    hipFree(changed);

    return 0;
}