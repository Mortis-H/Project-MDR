#include <iostream>
#include <hip/hip_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

// Constants
#define BLOCK_SIZE 256
#define NUM_NODES 128

// Kernel to initialize communities
__global__ void initializeCommunities(int *communities) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_NODES) {
        communities[idx] = idx; // Each node starts in its own community
    }
}

// Kernel to compute modularity gain
__global__ void computeModularityGain(const int *adjMatrix, const int *weights, const int *degrees, int *communities, float *modularityGain, float resolution) {
    extern __shared__ int sharedCommunities[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_NODES) return;

    int node = idx;
    int nodeCommunity = communities[node];
    int nodeDegree = degrees[node];
    float totalWeight = 0.0f;
    float internalWeight = 0.0f;

    // Load community of each node into shared memory
    if (threadIdx.x < NUM_NODES) {
        sharedCommunities[threadIdx.x] = communities[threadIdx.x];
    }
    __syncthreads();

    // Calculate total weight and internal weight
    for (int j = 0; j < NUM_NODES; ++j) {
        if (adjMatrix[node * NUM_NODES + j] > 0) {
            totalWeight += weights[node * NUM_NODES + j];
            if (sharedCommunities[j] == nodeCommunity) {
                internalWeight += weights[node * NUM_NODES + j];
            }
        }
    }

    // Calculate modularity gain for each community
    for (int c = 0; c < NUM_NODES; ++c) {
        if (c == nodeCommunity) continue;
        float gain = 0.0f;
        float communityWeight = 0.0f;
        for (int j = 0; j < NUM_NODES; ++j) {
            if (adjMatrix[node * NUM_NODES + j] > 0 && sharedCommunities[j] == c) {
                communityWeight += weights[node * NUM_NODES + j];
            }
        }
        gain = (2.0f * communityWeight / totalWeight) - (degrees[node] * degrees[c] * resolution / (2.0f * totalWeight * totalWeight));
        modularityGain[node * NUM_NODES + c] = gain;
    }
}

// Kernel to update communities
__global__ void updateCommunities(float *modularityGain, int *communities, int *bestCommunity) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_NODES) return;

    int node = idx;
    int currentCommunity = communities[node];
    float maxGain = 0.0f;
    int newCommunity = currentCommunity;

    // Find the community with the maximum modularity gain
    for (int c = 0; c < NUM_NODES; ++c) {
        if (c == currentCommunity) continue;
        float gain = modularityGain[node * NUM_NODES + c];
        if (gain > maxGain) {
            maxGain = gain;
            newCommunity = c;
        }
    }

    // Update community if a better one is found
    if (newCommunity != currentCommunity) {
        communities[node] = newCommunity;
        atomicExch(&bestCommunity[node], newCommunity);
    }
}

int main() {
    // Host data
    thrust::host_vector<int> h_adjMatrix(NUM_NODES * NUM_NODES, 0);
    thrust::host_vector<int> h_weights(NUM_NODES * NUM_NODES, 0);
    thrust::host_vector<int> h_degrees(NUM_NODES, 0);
    thrust::host_vector<int> h_communities(NUM_NODES, 0);
    thrust::host_vector<float> h_modularityGain(NUM_NODES * NUM_NODES, 0.0f);
    thrust::host_vector<int> h_bestCommunity(NUM_NODES, 0);

    // Initialize adjacency matrix and weights (example data)
    for (int i = 0; i < NUM_NODES; ++i) {
        for (int j = i + 1; j < NUM_NODES; ++j) {
            if (rand() % 10 < 3) { // Randomly connect nodes with 30% probability
                h_adjMatrix[i * NUM_NODES + j] = 1;
                h_adjMatrix[j * NUM_NODES + i] = 1;
                h_weights[i * NUM_NODES + j] = rand() % 10 + 1; // Random weight between 1 and 10
                h_weights[j * NUM_NODES + i] = h_weights[i * NUM_NODES + j];
                h_degrees[i] += h_weights[i * NUM_NODES + j];
                h_degrees[j] += h_weights[j * NUM_NODES + i];
            }
        }
    }

    // Device data
    thrust::device_vector<int> d_adjMatrix = h_adjMatrix;
    thrust::device_vector<int> d_weights = h_weights;
    thrust::device_vector<int> d_degrees = h_degrees;
    thrust::device_vector<int> d_communities = h_communities;
    thrust::device_vector<float> d_modularityGain = h_modularityGain;
    thrust::device_vector<int> d_bestCommunity = h_bestCommunity;

    // Initialize communities
    int numBlocks = (NUM_NODES + BLOCK_SIZE - 1) / BLOCK_SIZE;
    initializeCommunities<<<numBlocks, BLOCK_SIZE>>>(thrust::raw_pointer_cast(d_communities.data()));
    hipDeviceSynchronize();

    // Louvain algorithm parameters
    float resolution = 1.0f;
    bool changed = true;

    // Louvain algorithm main loop
    while (changed) {
        changed = false;

        // Compute modularity gain
        computeModularityGain<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(
            thrust::raw_pointer_cast(d_adjMatrix.data()),
            thrust::raw_pointer_cast(d_weights.data()),
            thrust::raw_pointer_cast(d_degrees.data()),
            thrust::raw_pointer_cast(d_communities.data()),
            thrust::raw_pointer_cast(d_modularityGain.data()),
            resolution);
        hipDeviceSynchronize();

        // Update communities
        thrust::fill(d_bestCommunity.begin(), d_bestCommunity.end(), -1);
        updateCommunities<<<numBlocks, BLOCK_SIZE>>>(
            thrust::raw_pointer_cast(d_modularityGain.data()),
            thrust::raw_pointer_cast(d_communities.data()),
            thrust::raw_pointer_cast(d_bestCommunity.data()));
        hipDeviceSynchronize();

        // Check for changes
        for (int i = 0; i < NUM_NODES; ++i) {
            if (h_communities[i] != d_bestCommunity[i]) {
                changed = true;
                break;
            }
        }
        h_communities = d_bestCommunity;
    }

    // Copy results back to host
    h_communities = d_communities;

    // Print final communities
    std::cout << "Final communities:" << std::endl;
    for (int i = 0; i < NUM_NODES; ++i) {
        std::cout << "Node " << i << ": Community " << h_communities[i] << std::endl;
    }

    return 0;
}