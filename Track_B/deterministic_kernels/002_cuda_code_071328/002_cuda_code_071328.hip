#include <iostream>
#include <hip/hip_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/remove.h>
#include <thrust/execution_policy.h>

#define NUM_NODES 128

// Kernel to find nodes with zero in-degree
__global__ void findZeroInDegreeNodes(int* inDegree, int* zeroInDegreeNodes, int* zeroInDegreeCount, int numNodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numNodes && inDegree[idx] == 0) {
        int pos = atomicAdd(zeroInDegreeCount, 1);
        zeroInDegreeNodes[pos] = idx;
    }
}

// Kernel to update in-degrees of neighbors
__global__ void updateInDegrees(int* adjacencyMatrix, int* inDegree, int* zeroInDegreeNodes, int zeroInDegreeCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < zeroInDegreeCount) {
        int node = zeroInDegreeNodes[idx];
        for (int j = 0; j < NUM_NODES; ++j) {
            if (adjacencyMatrix[node * NUM_NODES + j] == 1) {
                atomicSub(&inDegree[j], 1);
            }
        }
    }
}

int main() {
    // Host graph data
    thrust::host_vector<int> h_adjacencyMatrix(NUM_NODES * NUM_NODES, 0);
    thrust::host_vector<int> h_inDegree(NUM_NODES, 0);

    // Example graph: Randomly create a Directed Acyclic Graph (DAG)
    for (int i = 0; i < NUM_NODES; ++i) {
        for (int j = i + 1; j < NUM_NODES; ++j) {
            if (rand() % 2) { // Randomly add an edge
                h_adjacencyMatrix[i * NUM_NODES + j] = 1;
                h_inDegree[j]++;
            }
        }
    }

    // Device graph data
    thrust::device_vector<int> d_adjacencyMatrix = h_adjacencyMatrix;
    thrust::device_vector<int> d_inDegree = h_inDegree;
    thrust::device_vector<int> d_zeroInDegreeNodes(NUM_NODES);
    thrust::device_vector<int> d_topologicalSort(NUM_NODES);
    int h_zeroInDegreeCount = 0;

    // Device pointers
    int* d_adjacencyMatrixPtr = thrust::raw_pointer_cast(d_adjacencyMatrix.data());
    int* d_inDegreePtr = thrust::raw_pointer_cast(d_inDegree.data());
    int* d_zeroInDegreeNodesPtr = thrust::raw_pointer_cast(d_zeroInDegreeNodes.data());
    int* d_topologicalSortPtr = thrust::raw_pointer_cast(d_topologicalSort.data());
    int* d_zeroInDegreeCountPtr = &h_zeroInDegreeCount;

    // CUDA setup
    int blockSize = 256;
    int numBlocks = (NUM_NODES + blockSize - 1) / blockSize;

    // Topological sort
    int sortedCount = 0;
    while (sortedCount < NUM_NODES) {
        // Reset zero in-degree count
        hipMemset(d_zeroInDegreeCountPtr, 0, sizeof(int));

        // Find nodes with zero in-degree
        findZeroInDegreeNodes<<<numBlocks, blockSize>>>(d_inDegreePtr, d_zeroInDegreeNodesPtr, d_zeroInDegreeCountPtr, NUM_NODES);
        hipDeviceSynchronize();

        // Copy zero in-degree count back to host
        hipMemcpy(&h_zeroInDegreeCount, d_zeroInDegreeCountPtr, sizeof(int), hipMemcpyDeviceToHost);

        // If no zero in-degree nodes are found, graph has a cycle
        if (h_zeroInDegreeCount == 0) {
            std::cerr << "Graph has a cycle!" << std::endl;
            return -1;
        }

        // Update topological sort
        hipMemcpy(d_topologicalSortPtr + sortedCount, d_zeroInDegreeNodesPtr, h_zeroInDegreeCount * sizeof(int), hipMemcpyDeviceToDevice);
        sortedCount += h_zeroInDegreeCount;

        // Update in-degrees of neighbors
        updateInDegrees<<<(h_zeroInDegreeCount + blockSize - 1) / blockSize, blockSize>>>(d_adjacencyMatrixPtr, d_inDegreePtr, d_zeroInDegreeNodesPtr, h_zeroInDegreeCount);
        hipDeviceSynchronize();
    }

    // Copy result back to host
    thrust::host_vector<int> h_topologicalSort = d_topologicalSort;

    // Output topological sort
    std::cout << "Topological Sort: ";
    for (int i = 0; i < NUM_NODES; ++i) {
        std::cout << h_topologicalSort[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}