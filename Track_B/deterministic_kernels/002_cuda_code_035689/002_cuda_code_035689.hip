#include <iostream>
#include <hip/hip_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

// Constants
#define NUM_NODES 2048
#define MAX_EDGES 4096  // Assuming each node can connect to up to 2 neighbors on average

// Structure to represent an edge
struct Edge {
    int src;
    int dst;
    float weight;
};

// Kernel to initialize communities
__global__ void initCommunities(int* communities) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_NODES) {
        communities[idx] = idx;  // Each node starts in its own community
    }
}

// Kernel to update communities based on edge weights
__global__ void updateCommunities(Edge* edges, int* communities, int numEdges) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numEdges) {
        int srcComm = communities[edges[idx].src];
        int dstComm = communities[edges[idx].dst];
        if (srcComm != dstComm && edges[idx].weight > 0.5) {  // Threshold for merging communities
            communities[dstComm] = srcComm;  // Merge communities
        }
    }
}

int main() {
    // Initialize host data
    thrust::host_vector<Edge> h_edges(MAX_EDGES);
    for (int i = 0; i < MAX_EDGES; ++i) {
        h_edges[i].src = rand() % NUM_NODES;
        h_edges[i].dst = rand() % NUM_NODES;
        h_edges[i].weight = static_cast<float>(rand()) / RAND_MAX;
    }

    // Initialize device data
    thrust::device_vector<Edge> d_edges = h_edges;
    thrust::device_vector<int> d_communities(NUM_NODES);

    // Launch kernel to initialize communities
    int blockSize = 256;
    int numBlocks = (NUM_NODES + blockSize - 1) / blockSize;
    initCommunities<<<numBlocks, blockSize>>>(thrust::raw_pointer_cast(d_communities.data()));
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error in initCommunities kernel: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Launch kernel to update communities
    numBlocks = (MAX_EDGES + blockSize - 1) / blockSize;
    updateCommunities<<<numBlocks, blockSize>>>(thrust::raw_pointer_cast(d_edges.data()), thrust::raw_pointer_cast(d_communities.data()), MAX_EDGES);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error in updateCommunities kernel: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy results back to host
    thrust::host_vector<int> h_communities = d_communities;

    // Print the first 10 communities for demonstration
    std::cout << "First 10 communities after update:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << "Node " << i << " in community " << h_communities[i] << std::endl;
    }

    // Clean up
    hipDeviceReset();
    return 0;
}