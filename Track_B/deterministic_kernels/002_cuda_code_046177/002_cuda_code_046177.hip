#include <iostream>
#include <hip/hip_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>

#define NUM_NODES 128
#define MAX_EDGES 1024  // Assuming a sparse graph with at most 1024 edges

__global__ void initializeLabels(int *labels, int numNodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numNodes) {
        labels[idx] = idx;
    }
}

__global__ void propagateLabels(int *labels, int *edges, int *edgeList, int numEdges, int numNodes) {
    extern __shared__ int s_labels[];

    // Load labels into shared memory
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numNodes) {
        s_labels[idx] = labels[idx];
    }
    __syncthreads();

    // Propagate labels
    for (int i = threadIdx.x; i < numEdges; i += blockDim.x) {
        int u = edgeList[i];
        int v = edges[i];
        int labelU = s_labels[u];
        int labelV = s_labels[v];
        if (labelU < labelV) {
            atomicMin(&labels[v], labelU);
        }
        if (labelV < labelU) {
            atomicMin(&labels[u], labelV);
        }
    }
    __syncthreads();

    // Update shared memory labels
    if (idx < numNodes) {
        s_labels[idx] = labels[idx];
    }
    __syncthreads();

    // Second round of propagation
    for (int i = threadIdx.x; i < numEdges; i += blockDim.x) {
        int u = edgeList[i];
        int v = edges[i];
        int labelU = s_labels[u];
        int labelV = s_labels[v];
        if (labelU < labelV) {
            atomicMin(&labels[v], labelU);
        }
        if (labelV < labelU) {
            atomicMin(&labels[u], labelV);
        }
    }
}

int main() {
    // Example graph represented as an adjacency list
    int h_edges[MAX_EDGES] = { /* Initialize with edge data */ };
    int h_edgeList[MAX_EDGES] = { /* Initialize with edge list data */ };
    int numEdges = 256;  // Number of edges in the graph

    // Initialize labels on host
    thrust::host_vector<int> h_labels(NUM_NODES);
    for (int i = 0; i < NUM_NODES; ++i) {
        h_labels[i] = i;
    }

    // Allocate device memory
    int *d_labels, *d_edges, *d_edgeList;
    hipMalloc(&d_labels, NUM_NODES * sizeof(int));
    hipMalloc(&d_edges, numEdges * sizeof(int));
    hipMalloc(&d_edgeList, numEdges * sizeof(int));

    // Copy data to device
    hipMemcpy(d_labels, h_labels.data(), NUM_NODES * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_edges, h_edges, numEdges * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_edgeList, h_edgeList, numEdges * sizeof(int), hipMemcpyHostToDevice);

    // Launch kernel to initialize labels
    int blockSize = 256;
    int numBlocks = (NUM_NODES + blockSize - 1) / blockSize;
    initializeLabels<<<numBlocks, blockSize>>>(d_labels, NUM_NODES);

    // Kernel configuration
    int sharedMemSize = NUM_NODES * sizeof(int);
    numBlocks = (numEdges + blockSize - 1) / blockSize;

    // Launch kernel to propagate labels
    for (int iter = 0; iter < 10; ++iter) {  // Assuming a fixed number of iterations for simplicity
        propagateLabels<<<numBlocks, blockSize, sharedMemSize>>>(d_labels, d_edges, d_edgeList, numEdges, NUM_NODES);
        hipDeviceSynchronize();
        hipError_t err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
            return -1;
        }
    }

    // Copy labels back to host
    hipMemcpy(h_labels.data(), d_labels, NUM_NODES * sizeof(int), hipMemcpyDeviceToHost);

    // Output the labels (representing SCCs)
    for (int i = 0; i < NUM_NODES; ++i) {
        std::cout << "Node " << i << " has label " << h_labels[i] << std::endl;
    }

    // Free device memory
    hipFree(d_labels);
    hipFree(d_edges);
    hipFree(d_edgeList);

    return 0;
}