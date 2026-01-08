#include <iostream>
#include <hip/hip_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/remove.h>
#include <thrust/gather.h>
#include <thrust/extrema.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#define NUM_NODES 128
#define MAX_EDGES 1024
#define MAX_THREADS_PER_BLOCK 256

__global__ void computeInDegrees(int* edges, int* inDegrees, int numEdges) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numEdges) {
        atomicAdd(&inDegrees[edges[idx * 2 + 1]], 1);
    }
}

__global__ void findZeroInDegreeNodes(int* inDegrees, int* zeroDegreeNodes, int* zeroDegreeCount, int numNodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numNodes && inDegrees[idx] == 0) {
        int pos = atomicAdd(zeroDegreeCount, 1);
        zeroDegreeNodes[pos] = idx;
    }
}

__global__ void removeEdges(int* edges, int* inDegrees, int* zeroDegreeNodes, int* zeroDegreeCount, int numEdges) {
    extern __shared__ int sharedZeroDegreeNodes[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (tid < *zeroDegreeCount) {
        sharedZeroDegreeNodes[tid] = zeroDegreeNodes[tid];
    }
    __syncthreads();

    if (idx < numEdges) {
        int dest = edges[idx * 2 + 1];
        for (int i = 0; i < *zeroDegreeCount; ++i) {
            if (edges[idx * 2] == sharedZeroDegreeNodes[i]) {
                atomicSub(&inDegrees[dest], 1);
                edges[idx * 2] = -1; // Mark edge as removed
                break;
            }
        }
    }
}

__global__ void resetZeroDegreeNodes(int* zeroDegreeNodes, int* zeroDegreeCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < *zeroDegreeCount) {
        zeroDegreeNodes[idx] = -1;
    }
}

int main() {
    // Initialize host graph data
    thrust::host_vector<int> h_edges(MAX_EDGES * 2);
    thrust::host_vector<int> h_inDegrees(NUM_NODES, 0);
    thrust::host_vector<int> h_zeroDegreeNodes(NUM_NODES, -1);
    int h_zeroDegreeCount = 0;

    // Example graph initialization (randomly generated for demonstration)
    for (int i = 0; i < MAX_EDGES; ++i) {
        h_edges[i * 2] = rand() % NUM_NODES;
        h_edges[i * 2 + 1] = rand() % NUM_NODES;
    }

    // Allocate device memory
    int* d_edges;
    int* d_inDegrees;
    int* d_zeroDegreeNodes;
    int* d_zeroDegreeCount;
    hipMalloc(&d_edges, MAX_EDGES * 2 * sizeof(int));
    hipMalloc(&d_inDegrees, NUM_NODES * sizeof(int));
    hipMalloc(&d_zeroDegreeNodes, NUM_NODES * sizeof(int));
    hipMalloc(&d_zeroDegreeCount, sizeof(int));

    // Copy data to device
    hipMemcpy(d_edges, h_edges.data(), MAX_EDGES * 2 * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_inDegrees, h_inDegrees.data(), NUM_NODES * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_zeroDegreeNodes, h_zeroDegreeNodes.data(), NUM_NODES * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_zeroDegreeCount, &h_zeroDegreeCount, sizeof(int), hipMemcpyHostToDevice);

    // Compute in-degrees
    int numBlocks = (MAX_EDGES + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;
    computeInDegrees<<<numBlocks, MAX_THREADS_PER_BLOCK>>>(d_edges, d_inDegrees, MAX_EDGES);
    hipDeviceSynchronize();

    // Topological sort
    thrust::device_vector<int> d_topologicalSort;
    while (true) {
        // Find zero in-degree nodes
        thrust::fill(thrust::device, d_zeroDegreeNodes, d_zeroDegreeNodes + NUM_NODES, -1);
        thrust::fill(thrust::device, d_zeroDegreeCount, d_zeroDegreeCount + 1, 0);
        findZeroInDegreeNodes<<<numBlocks, MAX_THREADS_PER_BLOCK>>>(d_inDegrees, d_zeroDegreeNodes, d_zeroDegreeCount, NUM_NODES);
        hipDeviceSynchronize();

        // Copy zero-degree count to host
        hipMemcpy(&h_zeroDegreeCount, d_zeroDegreeCount, sizeof(int), hipMemcpyDeviceToHost);
        if (h_zeroDegreeCount == 0) break;

        // Copy zero-degree nodes to host
        h_zeroDegreeNodes.resize(h_zeroDegreeCount);
        hipMemcpy(h_zeroDegreeNodes.data(), d_zeroDegreeNodes, h_zeroDegreeCount * sizeof(int), hipMemcpyDeviceToHost);

        // Append zero-degree nodes to topological sort
        d_topologicalSort.insert(d_topologicalSort.end(), h_zeroDegreeNodes.begin(), h_zeroDegreeNodes.end());

        // Remove edges from zero-degree nodes
        removeEdges<<<numBlocks, MAX_THREADS_PER_BLOCK, h_zeroDegreeCount * sizeof(int)>>>(d_edges, d_inDegrees, d_zeroDegreeNodes, d_zeroDegreeCount, MAX_EDGES);
        hipDeviceSynchronize();

        // Reset zero-degree nodes
        resetZeroDegreeNodes<<<numBlocks, MAX_THREADS_PER_BLOCK>>>(d_zeroDegreeNodes, d_zeroDegreeCount);
        hipDeviceSynchronize();
    }

    // Copy topological sort result to host
    thrust::host_vector<int> h_topologicalSort = d_topologicalSort;

    // Print topological sort result
    std::cout << "Topological Sort: ";
    for (int i = 0; i < h_topologicalSort.size(); ++i) {
        std::cout << h_topologicalSort[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    hipFree(d_edges);
    hipFree(d_inDegrees);
    hipFree(d_zeroDegreeNodes);
    hipFree(d_zeroDegreeCount);

    return 0;
}