#include <iostream>
#include <hip/hip_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/equal.h>
#include <thrust/fill.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/extrema.h>

#define BLOCK_SIZE 256

// Kernel to perform edge contraction
__global__ void contractEdges(int* edges, int* labels, int numEdges) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numEdges) {
        int src = labels[edges[2 * tid]];
        int dst = labels[edges[2 * tid + 1]];
        if (src != dst) {
            edges[2 * tid] = src;
            edges[2 * tid + 1] = dst;
        }
    }
}

// Kernel to perform label propagation
__global__ void propagateLabels(int* edges, int* labels, int numEdges) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numEdges) {
        int src = edges[2 * tid];
        int dst = edges[2 * tid + 1];
        atomicMin(&labels[dst], labels[src]);
    }
}

// Kernel to find the root of a label
__global__ void findRoots(int* labels, int numNodes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numNodes) {
        while (labels[tid] != labels[labels[tid]]) {
            labels[tid] = labels[labels[tid]];
        }
    }
}

// Kernel to assign unique IDs to each SCC
__global__ void assignSCCIDs(int* labels, int* sccIDs, int numNodes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numNodes) {
        sccIDs[tid] = labels[tid];
    }
}

int main() {
    // Example graph with 64 nodes and edges
    const int numNodes = 64;
    const int numEdges = 128; // Example number of edges

    // Host data
    thrust::host_vector<int> h_edges(numEdges * 2);
    thrust::host_vector<int> h_labels(numNodes);
    thrust::host_vector<int> h_sccIDs(numNodes);

    // Initialize edges (example graph)
    for (int i = 0; i < numEdges; ++i) {
        h_edges[2 * i] = i % numNodes;
        h_edges[2 * i + 1] = (i + 1) % numNodes;
    }

    // Initialize labels
    thrust::sequence(h_labels.begin(), h_labels.end());

    // Device data
    thrust::device_vector<int> d_edges = h_edges;
    thrust::device_vector<int> d_labels = h_labels;
    thrust::device_vector<int> d_sccIDs = h_sccIDs;

    // Number of blocks and threads
    int numBlocks = (numEdges + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Perform edge contraction
    contractEdges<<<numBlocks, BLOCK_SIZE>>>(thrust::raw_pointer_cast(d_edges.data()),
                                             thrust::raw_pointer_cast(d_labels.data()),
                                             numEdges);
    hipDeviceSynchronize();

    // Sort edges to ensure they are ready for label propagation
    thrust::sort_by_key(d_edges.begin(), d_edges.end(), d_edges.begin() + numEdges);

    // Perform label propagation
    propagateLabels<<<numBlocks, BLOCK_SIZE>>>(thrust::raw_pointer_cast(d_edges.data()),
                                               thrust::raw_pointer_cast(d_labels.data()),
                                               numEdges);
    hipDeviceSynchronize();

    // Find roots of labels
    numBlocks = (numNodes + BLOCK_SIZE - 1) / BLOCK_SIZE;
    findRoots<<<numBlocks, BLOCK_SIZE>>>(thrust::raw_pointer_cast(d_labels.data()), numNodes);
    hipDeviceSynchronize();

    // Assign unique IDs to each SCC
    assignSCCIDs<<<numBlocks, BLOCK_SIZE>>>(thrust::raw_pointer_cast(d_labels.data()),
                                            thrust::raw_pointer_cast(d_sccIDs.data()),
                                            numNodes);
    hipDeviceSynchronize();

    // Copy results back to host
    thrust::copy(d_sccIDs.begin(), d_sccIDs.end(), h_sccIDs.begin());

    // Output the SCC IDs
    std::cout << "Strongly Connected Components IDs:\n";
    for (int i = 0; i < numNodes; ++i) {
        std::cout << "Node " << i << ": SCC " << h_sccIDs[i] << "\n";
    }

    // Check for errors
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return 1;
    }

    return 0;
}