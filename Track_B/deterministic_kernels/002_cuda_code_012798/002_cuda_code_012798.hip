#include <iostream>
#include <hip/hip_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

// Define the number of nodes
#define NUM_NODES 256

// Kernel to initialize community labels
__global__ void initCommunities(int *communities) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_NODES) {
        communities[idx] = idx; // Each node starts in its own community
    }
}

// Kernel to compute the change in modularity
__global__ void computeDeltaQ(int *adjMatrix, int *communities, float *deltaQ, int *degrees, int *nodeList, int numNodes) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node < numNodes) {
        int nodeCommunity = communities[node];
        int degree = degrees[node];
        float deltaQValue = 0.0f;

        // Calculate the change in modularity for each neighbor
        for (int neighbor = 0; neighbor < numNodes; ++neighbor) {
            if (adjMatrix[node * numNodes + neighbor] == 1) {
                int neighborCommunity = communities[neighbor];
                if (neighborCommunity == nodeCommunity) {
                    deltaQValue += 2.0f * adjMatrix[node * numNodes + neighbor] / numNodes - degree * degrees[neighbor] / (numNodes * numNodes);
                }
            }
        }
        deltaQ[node] = deltaQValue;
    }
}

// Main function to demonstrate the kernel's usage
int main() {
    // Initialize the adjacency matrix and other data structures
    thrust::host_vector<int> h_adjMatrix(NUM_NODES * NUM_NODES, 0);
    thrust::host_vector<int> h_degrees(NUM_NODES, 0);
    thrust::host_vector<int> h_communities(NUM_NODES, 0);
    thrust::host_vector<float> h_deltaQ(NUM_NODES, 0.0f);
    thrust::host_vector<int> h_nodeList(NUM_NODES);
    for (int i = 0; i < NUM_NODES; ++i) {
        h_nodeList[i] = i;
    }

    // Example: Create a simple ring graph
    for (int i = 0; i < NUM_NODES; ++i) {
        h_adjMatrix[i * NUM_NODES + (i + 1) % NUM_NODES] = 1;
        h_adjMatrix[(i + 1) % NUM_NODES * NUM_NODES + i] = 1;
        h_degrees[i] = 2;
    }

    // Allocate device memory
    int *d_adjMatrix, *d_communities, *d_degrees, *d_nodeList;
    float *d_deltaQ;
    hipMalloc(&d_adjMatrix, NUM_NODES * NUM_NODES * sizeof(int));
    hipMalloc(&d_communities, NUM_NODES * sizeof(int));
    hipMalloc(&d_degrees, NUM_NODES * sizeof(int));
    hipMalloc(&d_deltaQ, NUM_NODES * sizeof(float));
    hipMalloc(&d_nodeList, NUM_NODES * sizeof(int));

    // Copy data from host to device
    hipMemcpyAsync(d_adjMatrix, h_adjMatrix.data(), NUM_NODES * NUM_NODES * sizeof(int), hipMemcpyHostToDevice, 0);
    hipMemcpyAsync(d_degrees, h_degrees.data(), NUM_NODES * sizeof(int), hipMemcpyHostToDevice, 0);
    hipMemcpyAsync(d_nodeList, h_nodeList.data(), NUM_NODES * sizeof(int), hipMemcpyHostToDevice, 0);

    // Initialize communities
    int blockSize = 256;
    int numBlocks = (NUM_NODES + blockSize - 1) / blockSize;
    initCommunities<<<numBlocks, blockSize>>>(d_communities);
    hipDeviceSynchronize();

    // Compute deltaQ
    computeDeltaQ<<<numBlocks, blockSize>>>(d_adjMatrix, d_communities, d_deltaQ, d_degrees, d_nodeList, NUM_NODES);
    hipDeviceSynchronize();

    // Copy results back to host
    hipMemcpyAsync(h_deltaQ.data(), d_deltaQ, NUM_NODES * sizeof(float), hipMemcpyDeviceToHost, 0);
    hipMemcpyAsync(h_communities.data(), d_communities, NUM_NODES * sizeof(int), hipMemcpyDeviceToHost, 0);
    hipDeviceSynchronize();

    // Free device memory
    hipFree(d_adjMatrix);
    hipFree(d_communities);
    hipFree(d_degrees);
    hipFree(d_deltaQ);
    hipFree(d_nodeList);

    // Print results
    std::cout << "Final communities:" << std::endl;
    for (int i = 0; i < NUM_NODES; ++i) {
        std::cout << "Node " << i << ": Community " << h_communities[i] << std::endl;
    }

    return 0;
}