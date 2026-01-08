#include <iostream>
#include <hip/hip_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#define BLOCK_SIZE 32
#define NUM_NODES 32

// Structure to hold an edge with its betweenness centrality
struct Edge {
    int src;
    int dst;
    float betweenness;
};

// Kernel to calculate the betweenness centrality of edges
__global__ void calculateBetweennessKernel(int* edges, float* betweenness, int numEdges, int numNodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numEdges) return;

    int src = edges[idx * 2];
    int dst = edges[idx * 2 + 1];
    float localBetweenness = 0.0f;

    // Simple heuristic for betweenness calculation (not the full Girvan-Newman algorithm)
    // In a real implementation, you would need to perform BFS and backtracking to calculate betweenness
    localBetweenness = (float)(src + dst) / numNodes; // Dummy calculation

    betweenness[idx] = localBetweenness;
}

// Kernel to remove the edge with the highest betweenness centrality
__global__ void removeEdgeKernel(int* edges, float* betweenness, int* removedEdges, int numEdges, int* numRemovedEdges) {
    extern __shared__ float s_betweenness[];
    extern __shared__ int s_indices[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numEdges) return;

    s_betweenness[threadIdx.x] = betweenness[idx];
    s_indices[threadIdx.x] = idx;
    __syncthreads();

    // Use parallel reduction to find the maximum betweenness
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (s_betweenness[threadIdx.x] < s_betweenness[threadIdx.x + s]) {
                s_betweenness[threadIdx.x] = s_betweenness[threadIdx.x + s];
                s_indices[threadIdx.x] = s_indices[threadIdx.x + s];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicExch(&removedEdges[*numRemovedEdges], s_indices[0]);
        atomicAdd(numRemovedEdges, 1);
    }
}

int main() {
    // Example graph edges (undirected)
    int edges_h[] = {0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 29, 29, 30, 30, 31, 31, 0, 31, 1, 30, 2, 29, 3, 28, 4, 27, 5, 26, 6, 25, 7, 24, 8, 23, 9, 22, 10, 21, 11, 20, 12, 19, 13, 18, 14, 17, 15, 16};
    int numEdges = sizeof(edges_h) / (2 * sizeof(int));

    // Allocate device memory
    int* edges_d;
    float* betweenness_d;
    int* removedEdges_d;
    int* numRemovedEdges_d;

    hipMalloc(&edges_d, numEdges * 2 * sizeof(int));
    hipMalloc(&betweenness_d, numEdges * sizeof(float));
    hipMalloc(&removedEdges_d, numEdges * sizeof(int));
    hipMalloc(&numRemovedEdges_d, sizeof(int));

    // Copy data to device
    hipMemcpy(edges_d, edges_h, numEdges * 2 * sizeof(int), hipMemcpyHostToDevice);
    hipMemset(betweenness_d, 0, numEdges * sizeof(float));
    hipMemset(removedEdges_d, 0, numEdges * sizeof(int));
    hipMemset(numRemovedEdges_d, 0, sizeof(int));

    // Calculate betweenness centrality
    calculateBetweennessKernel<<<(numEdges + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(edges_d, betweenness_d, numEdges, NUM_NODES);
    hipDeviceSynchronize();

    // Remove edge with the highest betweenness centrality
    removeEdgeKernel<<<1, BLOCK_SIZE, BLOCK_SIZE * (sizeof(float) + sizeof(int))>>>(edges_d, betweenness_d, removedEdges_d, numEdges, numRemovedEdges_d);
    hipDeviceSynchronize();

    // Copy results back to host
    int numRemovedEdges_h;
    hipMemcpy(&numRemovedEdges_h, numRemovedEdges_d, sizeof(int), hipMemcpyDeviceToHost);
    int* removedEdges_h = new int[numRemovedEdges_h];
    hipMemcpy(removedEdges_h, removedEdges_d, numRemovedEdges_h * sizeof(int), hipMemcpyDeviceToHost);

    // Output the removed edges
    std::cout << "Removed edges (indices): ";
    for (int i = 0; i < numRemovedEdges_h; ++i) {
        std::cout << removedEdges_h[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    hipFree(edges_d);
    hipFree(betweenness_d);
    hipFree(removedEdges_d);
    hipFree(numRemovedEdges_d);

    delete[] removedEdges_h;

    return 0;
}