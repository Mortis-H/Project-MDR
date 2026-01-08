#include <iostream>
#include <hip/hip_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/equal.h>
#include <thrust/unique.h>
#include <thrust/execution_policy.h>

#define BLOCK_SIZE 256
#define MAX_NODES 512

__global__ void initializeLabels(int *labels, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        labels[idx] = idx;
    }
}

__global__ void propagateLabels(int *labels, int *adjMatrix, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        for (int j = 0; j < n; j++) {
            if (adjMatrix[idx * n + j] == 1) {
                labels[idx] = min(labels[idx], labels[j]);
            }
        }
    }
}

__global__ void propagateLabelsReverse(int *labels, int *adjMatrix, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        for (int j = 0; j < n; j++) {
            if (adjMatrix[j * n + idx] == 1) {
                labels[idx] = min(labels[idx], labels[j]);
            }
        }
    }
}

__global__ void assignLabels(int *labels, int *adjMatrix, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        for (int j = 0; j < n; j++) {
            if (adjMatrix[idx * n + j] == 1) {
                labels[j] = min(labels[j], labels[idx]);
            }
        }
    }
}

__global__ void assignLabelsReverse(int *labels, int *adjMatrix, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        for (int j = 0; j < n; j++) {
            if (adjMatrix[j * n + idx] == 1) {
                labels[j] = min(labels[j], labels[idx]);
            }
        }
    }
}

int main() {
    // Initialize the adjacency matrix for a directed graph with 512 nodes
    thrust::host_vector<int> h_adjMatrix(MAX_NODES * MAX_NODES, 0);
    // Example: create a simple directed graph
    h_adjMatrix[0 * MAX_NODES + 1] = 1;
    h_adjMatrix[1 * MAX_NODES + 2] = 1;
    h_adjMatrix[2 * MAX_NODES + 0] = 1;
    h_adjMatrix[3 * MAX_NODES + 4] = 1;
    h_adjMatrix[4 * MAX_NODES + 5] = 1;
    h_adjMatrix[5 * MAX_NODES + 3] = 1;

    thrust::device_vector<int> d_adjMatrix = h_adjMatrix;
    thrust::device_vector<int> d_labels(MAX_NODES);

    // Initialize labels
    initializeLabels<<<(MAX_NODES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(thrust::raw_pointer_cast(d_labels.data()), MAX_NODES);
    hipDeviceSynchronize();

    // Kernel launches for finding SCC
    for (int iter = 0; iter < 5; iter++) { // Perform multiple iterations for convergence
        propagateLabels<<<(MAX_NODES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(thrust::raw_pointer_cast(d_labels.data()), thrust::raw_pointer_cast(d_adjMatrix.data()), MAX_NODES);
        hipDeviceSynchronize();
        assignLabels<<<(MAX_NODES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(thrust::raw_pointer_cast(d_labels.data()), thrust::raw_pointer_cast(d_adjMatrix.data()), MAX_NODES);
        hipDeviceSynchronize();
        propagateLabelsReverse<<<(MAX_NODES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(thrust::raw_pointer_cast(d_labels.data()), thrust::raw_pointer_cast(d_adjMatrix.data()), MAX_NODES);
        hipDeviceSynchronize();
        assignLabelsReverse<<<(MAX_NODES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(thrust::raw_pointer_cast(d_labels.data()), thrust::raw_pointer_cast(d_adjMatrix.data()), MAX_NODES);
        hipDeviceSynchronize();
    }

    // Copy results back to host
    thrust::host_vector<int> h_labels = d_labels;

    // Output the labels
    std::cout << "Strongly Connected Components Labels:" << std::endl;
    for (int i = 0; i < MAX_NODES; i++) {
        std::cout << "Node " << i << ": Component " << h_labels[i] << std::endl;
    }

    return 0;
}