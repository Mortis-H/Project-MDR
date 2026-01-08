#include <iostream>
#include <hip/hip_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#define BLOCK_SIZE 256

__global__ void initializeCommunities(int *community, int numNodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numNodes) {
        community[idx] = idx; // Each node starts in its own community
    }
}

__global__ void computeModularityGain(int *adjMatrix, int *community, float *modularityGain, int *degrees, int numNodes) {
    extern __shared__ int sharedCommunity[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int node = idx / numNodes;
    int neighbor = idx % numNodes;

    if (idx < numNodes * numNodes) {
        int nodeCommunity = community[node];
        int neighborCommunity = community[neighbor];

        if (adjMatrix[idx] > 0) {
            atomicAdd(&sharedCommunity[nodeCommunity], adjMatrix[idx]);
        }

        if (threadIdx.x == 0) {
            sharedCommunity[nodeCommunity] = -sharedCommunity[nodeCommunity] + degrees[node];
        }
        __syncthreads();

        if (neighborCommunity != nodeCommunity && adjMatrix[idx] > 0) {
            modularityGain[idx] = 2.0f * adjMatrix[idx] - (float)degrees[node] * degrees[neighbor] / (2 * numNodes);
        }
    }
}

__global__ void updateCommunities(float *modularityGain, int *community, int numNodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numNodes) {
        int bestCommunity = idx;
        float bestGain = 0.0f;
        for (int i = 0; i < numNodes; ++i) {
            float gain = modularityGain[idx * numNodes + i];
            if (gain > bestGain) {
                bestGain = gain;
                bestCommunity = i;
            }
        }
        community[idx] = bestCommunity;
    }
}

int main() {
    const int numNodes = 1024;
    const int matrixSize = numNodes * numNodes;

    // Simulate an adjacency matrix for demonstration
    thrust::host_vector<int> h_adjMatrix(matrixSize, 0);
    thrust::host_vector<int> h_degrees(numNodes, 0);
    thrust::host_vector<int> h_community(numNodes, 0);

    // Randomly assign edges (this is a simple example, not a real graph)
    for (int i = 0; i < numNodes; ++i) {
        for (int j = 0; j < numNodes; ++j) {
            if (i != j && rand() % 100 < 5) { // 5% chance of an edge
                h_adjMatrix[i * numNodes + j] = 1;
                h_degrees[i]++;
            }
        }
    }

    // Allocate device memory
    int *d_adjMatrix, *d_community, *d_degrees;
    float *d_modularityGain;

    hipError_t err = hipMalloc((void **)&d_adjMatrix, matrixSize * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Error allocating d_adjMatrix: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    err = hipMalloc((void **)&d_degrees, numNodes * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Error allocating d_degrees: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    err = hipMalloc((void **)&d_community, numNodes * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Error allocating d_community: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    err = hipMalloc((void **)&d_modularityGain, matrixSize * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Error allocating d_modularityGain: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy data to device
    hipMemcpy(d_adjMatrix, h_adjMatrix.data(), matrixSize * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_degrees, h_degrees.data(), numNodes * sizeof(int), hipMemcpyHostToDevice);

    // Initialize communities
    int numBlocks = (numNodes + BLOCK_SIZE - 1) / BLOCK_SIZE;
    initializeCommunities<<<numBlocks, BLOCK_SIZE>>>(d_community, numNodes);
    hipDeviceSynchronize();

    // Compute modularity gain and update communities
    for (int iter = 0; iter < 10; ++iter) { // Run for 10 iterations for demonstration
        computeModularityGain<<<numBlocks, BLOCK_SIZE, numNodes * sizeof(int)>>>(d_adjMatrix, d_community, d_modularityGain, d_degrees, numNodes);
        hipDeviceSynchronize();

        updateCommunities<<<numBlocks, BLOCK_SIZE>>>(d_modularityGain, d_community, numNodes);
        hipDeviceSynchronize();
    }

    // Copy results back to host
    hipMemcpy(h_community.data(), d_community, numNodes * sizeof(int), hipMemcpyDeviceToHost);

    // Print communities (for demonstration purposes)
    for (int i = 0; i < 10; ++i) {
        std::cout << "Node " << i << " is in community " << h_community[i] << std::endl;
    }

    // Free device memory
    hipFree(d_adjMatrix);
    hipFree(d_degrees);
    hipFree(d_community);
    hipFree(d_modularityGain);

    return 0;
}