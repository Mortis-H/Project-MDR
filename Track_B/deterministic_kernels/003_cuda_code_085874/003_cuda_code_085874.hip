#include <iostream>
#include <hip/hip_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#define BLOCK_SIZE 256

// Kernel to compute the initial community assignment
__global__ void initializeCommunities(int *communities, int numNodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numNodes) {
        communities[idx] = idx; // Each node starts in its own community
    }
}

// Kernel to compute the total weight of edges inside each community
__global__ void computeCommunityWeights(int *communities, int *edgeList, float *weights, float *communityWeights, int numEdges) {
    extern __shared__ float sdata[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int edge = idx;
    if (edge < numEdges) {
        int src = edgeList[edge * 2];
        int dst = edgeList[edge * 2 + 1];
        int comm = communities[src];
        if (comm == communities[dst]) {
            atomicAdd(&sdata[threadIdx.x], weights[edge]);
        }
    }
    __syncthreads();

    // Reduce the partial sums in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            atomicAdd(&sdata[threadIdx.x], sdata[threadIdx.x + s]);
        }
        __syncthreads();
    }

    // Write the result for this block to global memory
    if (threadIdx.x == 0) {
        atomicAdd(&communityWeights[communities[edgeList[idx * 2]]], sdata[0]);
    }
}

// Kernel to compute the change in modularity for moving a node to a new community
__global__ void computeDeltaModularity(int *communities, int *edgeList, float *weights, float *totalWeight, float *communityWeights, float *deltaQ, int numEdges, int numNodes) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node < numNodes) {
        float currentCommWeight = communityWeights[communities[node]];
        float deltaQNode = 0.0f;
        for (int edge = 0; edge < numEdges; edge++) {
            int src = edgeList[edge * 2];
            int dst = edgeList[edge * 2 + 1];
            if (src == node) {
                int neighborComm = communities[dst];
                float edgeWeight = weights[edge];
                deltaQNode += 2 * edgeWeight - (currentCommWeight + 2 * edgeWeight) * (2 * edgeWeight) / (*totalWeight);
            } else if (dst == node) {
                int neighborComm = communities[src];
                float edgeWeight = weights[edge];
                deltaQNode += 2 * edgeWeight - (currentCommWeight + 2 * edgeWeight) * (2 * edgeWeight) / (*totalWeight);
            }
        }
        deltaQ[node] = deltaQNode;
    }
}

int main() {
    const int numNodes = 64;
    const int numEdges = 128; // Example number of edges

    // Host data
    thrust::host_vector<int> h_edgeList(numEdges * 2);
    thrust::host_vector<float> h_weights(numEdges);
    thrust::host_vector<int> h_communities(numNodes);
    thrust::host_vector<float> h_communityWeights(numNodes, 0.0f);
    thrust::host_vector<float> h_deltaQ(numNodes);

    // Initialize edge list and weights (example data)
    for (int i = 0; i < numEdges; ++i) {
        h_edgeList[i * 2] = i % numNodes;
        h_edgeList[i * 2 + 1] = (i + 1) % numNodes;
        h_weights[i] = 1.0f; // Example weights
    }

    // Device data
    thrust::device_vector<int> d_edgeList = h_edgeList;
    thrust::device_vector<float> d_weights = h_weights;
    thrust::device_vector<int> d_communities(numNodes);
    thrust::device_vector<float> d_communityWeights(numNodes);
    thrust::device_vector<float> d_deltaQ(numNodes);

    // Initialize communities
    initializeCommunities<<<(numNodes + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(thrust::raw_pointer_cast(d_communities.data()), numNodes);
    hipDeviceSynchronize();
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error in initializeCommunities: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Compute total weight of all edges
    float totalWeight = thrust::reduce(thrust::device, d_weights.begin(), d_weights.end(), 0.0f, thrust::plus<float>());
    float *d_totalWeight;
    hipMalloc(&d_totalWeight, sizeof(float));
    hipMemcpy(d_totalWeight, &totalWeight, sizeof(float), hipMemcpyHostToDevice);

    // Compute community weights
    computeCommunityWeights<<<(numEdges + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
        thrust::raw_pointer_cast(d_communities.data()),
        thrust::raw_pointer_cast(d_edgeList.data()),
        thrust::raw_pointer_cast(d_weights.data()),
        thrust::raw_pointer_cast(d_communityWeights.data()),
        numEdges
    );
    hipDeviceSynchronize();
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error in computeCommunityWeights: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Compute delta modularity for each node
    computeDeltaModularity<<<(numNodes + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        thrust::raw_pointer_cast(d_communities.data()),
        thrust::raw_pointer_cast(d_edgeList.data()),
        thrust::raw_pointer_cast(d_weights.data()),
        d_totalWeight,
        thrust::raw_pointer_cast(d_communityWeights.data()),
        thrust::raw_pointer_cast(d_deltaQ.data()),
        numEdges,
        numNodes
    );
    hipDeviceSynchronize();
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error in computeDeltaModularity: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy results back to host
    thrust::copy(d_communities.begin(), d_communities.end(), h_communities.begin());
    thrust::copy(d_deltaQ.begin(), d_deltaQ.end(), h_deltaQ.begin());

    // Print communities and deltaQ for each node
    std::cout << "Communities:" << std::endl;
    for (int i = 0; i < numNodes; ++i) {
        std::cout << "Node " << i << ": Community " << h_communities[i] << ", DeltaQ: " << h_deltaQ[i] << std::endl;
    }

    // Free device memory
    hipFree(d_totalWeight);

    return 0;
}