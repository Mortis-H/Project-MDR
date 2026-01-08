#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/remove.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#define BLOCK_SIZE 256

// Kernel to initialize community labels
__global__ void initializeCommunities(int *communities, int numNodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numNodes) {
        communities[idx] = idx; // Each node starts in its own community
    }
}

// Kernel to update community labels based on InfoMap algorithm
__global__ void updateCommunities(int *communities, int *newCommunities, int *edges, int *edgeWeights, int *nodeDegrees, int numEdges) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numEdges) {
        int src = edges[2 * idx];
        int dst = edges[2 * idx + 1];
        int srcComm = communities[src];
        int dstComm = communities[dst];
        int weight = edgeWeights[idx];

        // Simple rule: join the community with higher degree
        if (nodeDegrees[srcComm] < nodeDegrees[dstComm]) {
            newCommunities[src] = dstComm;
        } else if (nodeDegrees[srcComm] > nodeDegrees[dstComm]) {
            newCommunities[dst] = srcComm;
        }
    }
}

// Kernel to update community labels to their new values
__global__ void applyNewCommunities(int *communities, int *newCommunities, int numNodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numNodes) {
        communities[idx] = newCommunities[idx];
    }
}

// Kernel to calculate node degrees
__global__ void calculateDegrees(int *degrees, int *edges, int numEdges) {
    extern __shared__ int sdata[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Initialize shared memory
    sdata[tid] = 0;
    __syncthreads();

    // Accumulate degrees in shared memory
    if (idx < numEdges) {
        atomicAdd(&sdata[0], 1); // Each edge contributes to the degree of two nodes
    }
    __syncthreads();

    // Write result for this block to global memory
    if (tid == 0) {
        atomicAdd(&degrees[blockIdx.x], sdata[0]);
    }
}

int main() {
    const int numNodes = 16384;
    const int numEdges = 32768; // Example number of edges

    // Host data
    thrust::host_vector<int> h_edges(2 * numEdges);
    thrust::host_vector<int> h_edgeWeights(numEdges);
    thrust::host_vector<int> h_communities(numNodes);
    thrust::host_vector<int> h_newCommunities(numNodes);
    thrust::host_vector<int> h_nodeDegrees(numNodes, 0);

    // Initialize edges and weights (example data)
    for (int i = 0; i < numEdges; ++i) {
        h_edges[2 * i] = i % numNodes;
        h_edges[2 * i + 1] = (i + 1) % numNodes;
        h_edgeWeights[i] = 1; // Uniform weights for simplicity
    }

    // Device data
    thrust::device_vector<int> d_edges = h_edges;
    thrust::device_vector<int> d_edgeWeights = h_edgeWeights;
    thrust::device_vector<int> d_communities(numNodes);
    thrust::device_vector<int> d_newCommunities(numNodes);
    thrust::device_vector<int> d_nodeDegrees(numNodes, 0);

    // Initialize communities
    initializeCommunities<<<(numNodes + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(thrust::raw_pointer_cast(d_communities.data()), numNodes);
    hipDeviceSynchronize();

    // Calculate node degrees
    calculateDegrees<<<(numNodes + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(thrust::raw_pointer_cast(d_nodeDegrees.data()), thrust::raw_pointer_cast(d_edges.data()), numEdges);
    hipDeviceSynchronize();

    // Perform a few iterations of community updates
    for (int iter = 0; iter < 10; ++iter) {
        // Update communities
        updateCommunities<<<(numEdges + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(thrust::raw_pointer_cast(d_communities.data()), thrust::raw_pointer_cast(d_newCommunities.data()), thrust::raw_pointer_cast(d_edges.data()), thrust::raw_pointer_cast(d_edgeWeights.data()), thrust::raw_pointer_cast(d_nodeDegrees.data()), numEdges);
        hipDeviceSynchronize();

        // Apply new communities
        applyNewCommunities<<<(numNodes + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(thrust::raw_pointer_cast(d_communities.data()), thrust::raw_pointer_cast(d_newCommunities.data()), numNodes);
        hipDeviceSynchronize();
    }

    // Copy results back to host
    thrust::copy(d_communities.begin(), d_communities.end(), h_communities.begin());

    // Check for errors
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return 1;
    }

    // Output some results for verification
    for (int i = 0; i < 10; ++i) {
        std::cout << "Node " << i << " community: " << h_communities[i] << std::endl;
    }

    return 0;
}