#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/sequence.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>

namespace cg = cooperative_groups;

#define CHECK(call)                                                            \
{                                                                              \
    const hipError_t error = call;                                            \
    if (error != hipSuccess)                                                  \
    {                                                                          \
        std::cerr << "Error: " << hipGetErrorString(error) << ":\n";          \
        std::cerr << __FILE__ << ":" << __LINE__ << std::endl;                   \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
}

// Define the number of nodes and edges
const int NUM_NODES = 64;
const int NUM_EDGES = 128;

// Kernel to initialize community labels
__global__ void initCommunities(int *communities) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_NODES) {
        communities[idx] = idx; // Each node starts in its own community
    }
}

// Kernel to calculate the total weight of edges in the graph
__global__ void calculateTotalWeight(int *weights, int *totalWeight) {
    extern __shared__ int sdata[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load shared memory
    if (idx < NUM_EDGES) {
        sdata[tid] = weights[idx];
    } else {
        sdata[tid] = 0;
    }
    __syncthreads();

    // Reduce in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) {
        atomicAdd(totalWeight, sdata[0]);
    }
}

// Kernel to compute the modularity gain
__global__ void computeModularityGain(int *weights, int *degrees, int *communities, float *modularityGain, int *totalWeight) {
    cg::grid_group grid = cg::this_grid();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_NODES) return;

    int node = idx;
    int currentCommunity = communities[node];
    float k_i = degrees[node];
    float m = *totalWeight;

    float deltaQ = 0.0f;
    float k_i_in = 0.0f;

    // Calculate k_i_in for the current community
    for (int e = 0; e < NUM_EDGES; ++e) {
        int src = e / NUM_NODES;
        int dst = e % NUM_NODES;
        if ((src == node && communities[dst] == currentCommunity) || (dst == node && communities[src] == currentCommunity)) {
            k_i_in += weights[e];
        }
    }

    // Calculate modularity gain for moving node to its own community
    deltaQ = (2.0f * k_i_in / m) - (k_i * k_i / (m * m));

    // Write the modularity gain for this node
    modularityGain[node] = deltaQ;
}

int main() {
    // Initialize host data
    thrust::host_vector<int> h_weights(NUM_EDGES);
    thrust::host_vector<int> h_degrees(NUM_NODES);
    thrust::host_vector<int> h_communities(NUM_NODES);
    thrust::host_vector<float> h_modularityGain(NUM_NODES);
    int h_totalWeight = 0;

    // Example graph weights (random values for demonstration)
    thrust::sequence(h_weights.begin(), h_weights.end());
    thrust::fill(h_degrees.begin(), h_degrees.end(), 4); // Example degrees
    thrust::fill(h_communities.begin(), h_communities.end(), 0); // Initialize communities

    // Copy data to device
    thrust::device_vector<int> d_weights = h_weights;
    thrust::device_vector<int> d_degrees = h_degrees;
    thrust::device_vector<int> d_communities = h_communities;
    thrust::device_vector<float> d_modularityGain = h_modularityGain;
    int *d_totalWeight;
    CHECK(hipMalloc(&d_totalWeight, sizeof(int)));

    // Initialize communities
    int blockSize = 256;
    int numBlocks = (NUM_NODES + blockSize - 1) / blockSize;
    initCommunities<<<numBlocks, blockSize>>>(thrust::raw_pointer_cast(d_communities.data()));
    CHECK(hipGetLastError());

    // Calculate total weight of edges
    numBlocks = (NUM_EDGES + blockSize - 1) / blockSize;
    calculateTotalWeight<<<numBlocks, blockSize, blockSize * sizeof(int)>>>(thrust::raw_pointer_cast(d_weights.data()), d_totalWeight);
    CHECK(hipGetLastError());
    CHECK(hipMemcpy(&h_totalWeight, d_totalWeight, sizeof(int), hipMemcpyDeviceToHost));

    // Compute modularity gain
    numBlocks = (NUM_NODES + blockSize - 1) / blockSize;
    computeModularityGain<<<numBlocks, blockSize>>>(thrust::raw_pointer_cast(d_weights.data()),
                                                  thrust::raw_pointer_cast(d_degrees.data()),
                                                  thrust::raw_pointer_cast(d_communities.data()),
                                                  thrust::raw_pointer_cast(d_modularityGain.data()),
                                                  d_totalWeight);
    CHECK(hipGetLastError());

    // Copy results back to host
    thrust::copy(d_communities.begin(), d_communities.end(), h_communities.begin());
    thrust::copy(d_modularityGain.begin(), d_modularityGain.end(), h_modularityGain.begin());

    // Free device memory
    hipFree(d_totalWeight);

    // Output results
    std::cout << "Final communities: ";
    for (int i = 0; i < NUM_NODES; ++i) {
        std::cout << h_communities[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Modularity gains: ";
    for (int i = 0; i < NUM_NODES; ++i) {
        std::cout << h_modularityGain[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}