#include <iostream>
#include <hip/hip_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#define NUM_NODES 256
#define BLOCK_SIZE 256

// Function to check CUDA errors
#define CUDA_CHECK(call)                                                            \
    {                                                                               \
        hipError_t err = call;                                                     \
        if (err != hipSuccess) {                                                   \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - "    \
                      << hipGetErrorString(err) << std::endl;                      \
            exit(EXIT_FAILURE);                                                     \
        }                                                                           \
    }

// Kernel to calculate betweenness centrality (simplified version)
__global__ void calculateBetweenness(int* edges, int* edgeWeights, float* betweenness, int numEdges) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numEdges) return;

    // Simplified betweenness calculation
    betweenness[tid] += edgeWeights[tid] * 0.5f; // Placeholder for actual betweenness calculation
}

// Kernel to remove edge with highest betweenness
__global__ void removeEdge(int* edges, float* betweenness, int numEdges, int edgeToRemove) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numEdges) return;

    if (tid == edgeToRemove) {
        edges[tid] = -1; // Mark edge as removed
    }
}

int main() {
    // Example graph data (edges and weights)
    int edges[NUM_NODES * 2]; // Assuming each node can have up to 2 edges for simplicity
    int edgeWeights[NUM_NODES * 2];
    float betweenness[NUM_NODES * 2];

    // Initialize edges and weights (random example)
    for (int i = 0; i < NUM_NODES * 2; ++i) {
        edges[i] = i % NUM_NODES; // Placeholder for actual edge data
        edgeWeights[i] = rand() % 10 + 1; // Random weights between 1 and 10
        betweenness[i] = 0.0f;
    }

    // Allocate device memory
    int* d_edges;
    int* d_edgeWeights;
    float* d_betweenness;
    CUDA_CHECK(hipMalloc((void**)&d_edges, sizeof(int) * NUM_NODES * 2));
    CUDA_CHECK(hipMalloc((void**)&d_edgeWeights, sizeof(int) * NUM_NODES * 2));
    CUDA_CHECK(hipMalloc((void**)&d_betweenness, sizeof(float) * NUM_NODES * 2));

    // Asynchronous memory copy to device
    hipStream_t stream;
    CUDA_CHECK(hipStreamCreate(&stream));
    CUDA_CHECK(hipMemcpyAsync(d_edges, edges, sizeof(int) * NUM_NODES * 2, hipMemcpyHostToDevice, stream));
    CUDA_CHECK(hipMemcpyAsync(d_edgeWeights, edgeWeights, sizeof(int) * NUM_NODES * 2, hipMemcpyHostToDevice, stream));

    // Calculate betweenness centrality
    calculateBetweenness<<<(NUM_NODES * 2 + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(d_edges, d_edgeWeights, d_betweenness, NUM_NODES * 2);
    CUDA_CHECK(hipGetLastError());

    // Synchronize stream to ensure calculation is complete before copying back
    CUDA_CHECK(hipStreamSynchronize(stream));

    // Copy betweenness back to host
    CUDA_CHECK(hipMemcpyAsync(betweenness, d_betweenness, sizeof(float) * NUM_NODES * 2, hipMemcpyDeviceToHost, stream));

    // Find edge with highest betweenness
    float maxBetweenness = thrust::reduce(thrust::device, d_betweenness, d_betweenness + NUM_NODES * 2, 0.0f, thrust::maximum<float>());
    int edgeToRemove = thrust::max_element(thrust::device, d_betweenness, d_betweenness + NUM_NODES * 2) - d_betweenness;

    // Remove edge with highest betweenness
    removeEdge<<<(NUM_NODES * 2 + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(d_edges, d_betweenness, NUM_NODES * 2, edgeToRemove);
    CUDA_CHECK(hipGetLastError());

    // Synchronize stream to ensure removal is complete before copying back
    CUDA_CHECK(hipStreamSynchronize(stream));

    // Copy edges back to host
    CUDA_CHECK(hipMemcpyAsync(edges, d_edges, sizeof(int) * NUM_NODES * 2, hipMemcpyDeviceToHost, stream));

    // Clean up
    CUDA_CHECK(hipFree(d_edges));
    CUDA_CHECK(hipFree(d_edgeWeights));
    CUDA_CHECK(hipFree(d_betweenness));
    CUDA_CHECK(hipStreamDestroy(stream));

    // Output result (for demonstration purposes)
    std::cout << "Edge removed: " << edgeToRemove << " with betweenness: " << maxBetweenness << std::endl;

    return 0;
}