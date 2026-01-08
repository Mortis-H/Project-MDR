#include <iostream>
#include <hip/hip_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/gather.h>
#include <thrust/remove.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>

#define BLOCK_SIZE 256
#define NUM_GPUS 2

__global__ void computeBetweennessKernel(int* edges, int* weights, int* edgeCount, int* betweenness, int* communities, int numNodes, int numEdges) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numEdges) return;

    int u = edges[2 * tid];
    int v = edges[2 * tid + 1];
    int weight = weights[tid];

    // Simple betweenness calculation for demonstration
    // In practice, use Brandes' algorithm for more accuracy
    atomicAdd(&betweenness[u], weight);
    atomicAdd(&betweenness[v], weight);
}

__global__ void updateCommunitiesKernel(int* betweenness, int* communities, int numNodes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numNodes) return;

    // Simple community update logic
    // In practice, use a more sophisticated method
    if (betweenness[tid] > 10) {
        communities[tid] = 1;
    } else {
        communities[tid] = 0;
    }
}

void checkCudaError(hipError_t err, const char* msg) {
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << msg << " - " << hipGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void initializeGraph(int* edges, int* weights, int* edgeCount, int numNodes, int numEdges) {
    // Initialize graph edges and weights for demonstration
    // In practice, load graph from file or other source
    for (int i = 0; i < numEdges; ++i) {
        edges[2 * i] = i % numNodes;
        edges[2 * i + 1] = (i + 1) % numNodes;
        weights[i] = 1;
        edgeCount[i] = 1;
    }
}

int main() {
    int numNodes = 64;
    int numEdges = 128; // Fully connected graph for demonstration

    // Allocate host memory
    int* h_edges = new int[2 * numEdges];
    int* h_weights = new int[numEdges];
    int* h_edgeCount = new int[numEdges];
    int* h_betweenness = new int[numNodes];
    int* h_communities = new int[numNodes];

    // Initialize graph
    initializeGraph(h_edges, h_weights, h_edgeCount, numNodes, numEdges);
    thrust::fill_n(h_betweenness, numNodes, 0);
    thrust::fill_n(h_communities, numNodes, 0);

    // Allocate device memory
    int* d_edges[NUM_GPUS];
    int* d_weights[NUM_GPUS];
    int* d_edgeCount[NUM_GPUS];
    int* d_betweenness[NUM_GPUS];
    int* d_communities[NUM_GPUS];

    for (int gpu = 0; gpu < NUM_GPUS; ++gpu) {
        hipSetDevice(gpu);
        hipMalloc(&d_edges[gpu], 2 * numEdges * sizeof(int));
        hipMalloc(&d_weights[gpu], numEdges * sizeof(int));
        hipMalloc(&d_edgeCount[gpu], numEdges * sizeof(int));
        hipMalloc(&d_betweenness[gpu], numNodes * sizeof(int));
        hipMalloc(&d_communities[gpu], numNodes * sizeof(int));

        hipMemcpy(d_edges[gpu], h_edges, 2 * numEdges * sizeof(int), hipMemcpyHostToDevice);
        hipMemcpy(d_weights[gpu], h_weights, numEdges * sizeof(int), hipMemcpyHostToDevice);
        hipMemcpy(d_edgeCount[gpu], h_edgeCount, numEdges * sizeof(int), hipMemcpyHostToDevice);
        hipMemcpy(d_betweenness[gpu], h_betweenness, numNodes * sizeof(int), hipMemcpyHostToDevice);
        hipMemcpy(d_communities[gpu], h_communities, numNodes * sizeof(int), hipMemcpyHostToDevice);
    }

    // Launch kernels on each GPU
    for (int gpu = 0; gpu < NUM_GPUS; ++gpu) {
        hipSetDevice(gpu);
        int numBlocks = (numEdges + BLOCK_SIZE - 1) / BLOCK_SIZE;
        computeBetweennessKernel<<<numBlocks, BLOCK_SIZE>>>(d_edges[gpu], d_weights[gpu], d_edgeCount[gpu], d_betweenness[gpu], d_communities[gpu], numNodes, numEdges);
        checkCudaError(hipGetLastError(), "computeBetweennessKernel launch failed");

        numBlocks = (numNodes + BLOCK_SIZE - 1) / BLOCK_SIZE;
        updateCommunitiesKernel<<<numBlocks, BLOCK_SIZE>>>(d_betweenness[gpu], d_communities[gpu], numNodes);
        checkCudaError(hipGetLastError(), "updateCommunitiesKernel launch failed");
    }

    // Copy results back to host
    for (int gpu = 0; gpu < NUM_GPUS; ++gpu) {
        hipSetDevice(gpu);
        hipMemcpy(h_betweenness, d_betweenness[gpu], numNodes * sizeof(int), hipMemcpyDeviceToHost);
        hipMemcpy(h_communities, d_communities[gpu], numNodes * sizeof(int), hipMemcpyDeviceToHost);
    }

    // Free device memory
    for (int gpu = 0; gpu < NUM_GPUS; ++gpu) {
        hipSetDevice(gpu);
        hipFree(d_edges[gpu]);
        hipFree(d_weights[gpu]);
        hipFree(d_edgeCount[gpu]);
        hipFree(d_betweenness[gpu]);
        hipFree(d_communities[gpu]);
    }

    // Free host memory
    delete[] h_edges;
    delete[] h_weights;
    delete[] h_edgeCount;
    delete[] h_betweenness;
    delete[] h_communities;

    std::cout << "Community detection complete." << std::endl;
    return 0;
}