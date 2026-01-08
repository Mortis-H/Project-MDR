#include <iostream>
#include <hip/hip_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/scan.h>
#include <thrust/adjacent_difference.h>
#include <thrust/execution_policy.h>

#define BLOCK_SIZE 256
#define NUM_NODES 8192

// Kernel to initialize labels
__global__ void initializeLabels(int* labels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_NODES) {
        labels[idx] = idx;
    }
}

// Kernel to perform label propagation
__global__ void propagateLabels(int* edges, int* labels, int* newLabels, int* numEdges) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < *numEdges) {
        int u = edges[idx * 2];
        int v = edges[idx * 2 + 1];
        int labelU = labels[u];
        int labelV = labels[v];
        if (labelU < labelV) {
            atomicMin(&newLabels[v], labelU);
        }
    }
}

// Kernel to update labels
__global__ void updateLabels(int* labels, int* newLabels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_NODES) {
        if (newLabels[idx] < labels[idx]) {
            labels[idx] = newLabels[idx];
            newLabels[idx] = idx;
        }
    }
}

// Function to check for errors in CUDA calls
void checkCudaError(hipError_t err, const char* msg) {
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << msg << " (" << hipGetErrorString(err) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Example graph edges (source, destination)
    thrust::host_vector<int> h_edges = {
        // Add your graph edges here in the format: source1, destination1, source2, destination2, ...
        0, 1, 1, 2, 2, 0, 3, 4, 4, 5, 5, 3, 6, 7, 7, 8, 8, 6, 0, 3, 3, 6
    };
    int numEdges = h_edges.size() / 2;

    // Allocate device memory
    int* d_edges;
    int* d_labels;
    int* d_newLabels;
    int* d_numEdges;

    checkCudaError(hipMalloc(&d_edges, numEdges * 2 * sizeof(int)), "Failed to allocate d_edges");
    checkCudaError(hipMalloc(&d_labels, NUM_NODES * sizeof(int)), "Failed to allocate d_labels");
    checkCudaError(hipMalloc(&d_newLabels, NUM_NODES * sizeof(int)), "Failed to allocate d_newLabels");
    checkCudaError(hipMalloc(&d_numEdges, sizeof(int)), "Failed to allocate d_numEdges");

    // Copy data to device
    checkCudaError(hipMemcpy(d_edges, h_edges.data(), numEdges * 2 * sizeof(int), hipMemcpyHostToDevice), "Failed to copy edges to device");
    checkCudaError(hipMemcpy(d_numEdges, &numEdges, sizeof(int), hipMemcpyHostToDevice), "Failed to copy numEdges to device");

    // Initialize labels
    int numBlocks = (NUM_NODES + BLOCK_SIZE - 1) / BLOCK_SIZE;
    initializeLabels<<<numBlocks, BLOCK_SIZE>>>(d_labels);
    checkCudaError(hipGetLastError(), "Kernel launch failed: initializeLabels");

    // Main loop for label propagation
    bool changed = true;
    while (changed) {
        // Initialize new labels to current labels
        checkCudaError(hipMemcpy(d_newLabels, d_labels, NUM_NODES * sizeof(int), hipMemcpyDeviceToDevice), "Failed to copy labels to newLabels");

        // Propagate labels
        propagateLabels<<<numBlocks, BLOCK_SIZE>>>(d_edges, d_labels, d_newLabels, d_numEdges);
        checkCudaError(hipGetLastError(), "Kernel launch failed: propagateLabels");

        // Update labels
        updateLabels<<<numBlocks, BLOCK_SIZE>>>(d_labels, d_newLabels);
        checkCudaError(hipGetLastError(), "Kernel launch failed: updateLabels");

        // Check if labels have changed
        thrust::device_vector<int> d_labels_copy(d_labels, d_labels + NUM_NODES);
        thrust::device_vector<int> d_newLabels_copy(d_newLabels, d_newLabels + NUM_NODES);
        changed = !thrust::equal(d_labels_copy.begin(), d_labels_copy.end(), d_newLabels_copy.begin());
    }

    // Copy labels back to host
    thrust::host_vector<int> h_labels(NUM_NODES);
    checkCudaError(hipMemcpy(h_labels.data(), d_labels, NUM_NODES * sizeof(int), hipMemcpyDeviceToHost), "Failed to copy labels back to host");

    // Print labels (for demonstration purposes)
    std::cout << "Labels for each node:" << std::endl;
    for (int i = 0; i < NUM_NODES; ++i) {
        std::cout << "Node " << i << ": Label " << h_labels[i] << std::endl;
    }

    // Free device memory
    hipFree(d_edges);
    hipFree(d_labels);
    hipFree(d_newLabels);
    hipFree(d_numEdges);

    return 0;
}