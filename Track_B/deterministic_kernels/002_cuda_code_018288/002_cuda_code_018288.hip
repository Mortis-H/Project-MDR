#include <iostream>
#include <hip/hip_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/adjacent_difference.h>
#include <thrust/execution_policy.h>

// Define the number of nodes
#define NUM_NODES 4096

// Simple edge structure for the graph
struct Edge {
    int src;
    int dst;
    float weight;
};

// Kernel to initialize community assignments
__global__ void initializeCommunities(int *communities) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_NODES) {
        communities[idx] = idx; // Each node starts in its own community
    }
}

// Kernel to compute the modularity gain for each node
__global__ void computeModularityGain(const Edge *edges, const int *degrees, int *communities, float *modularityGain, int numEdges) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numEdges) {
        int src = edges[idx].src;
        int dst = edges[idx].dst;
        float weight = edges[idx].weight;

        // Calculate the modularity gain for src moving to dst's community
        int srcComm = communities[src];
        int dstComm = communities[dst];
        if (srcComm != dstComm) {
            atomicAdd(&modularityGain[src], weight - degrees[src] * degrees[dst] / (2.0f * NUM_NODES));
        }
    }
}

// Main function to demonstrate the usage of the kernels
int main() {
    // Example graph with edges and weights
    Edge h_edges[] = {
        {0, 1, 1.0f}, {0, 2, 1.0f}, {1, 2, 1.0f}, {1, 3, 1.0f}, {2, 3, 1.0f},
        // ... Add more edges to form a complete graph with 4096 nodes
    };
    int numEdges = sizeof(h_edges) / sizeof(h_edges[0]);

    // Allocate device memory
    Edge *d_edges;
    int *d_degrees, *d_communities;
    float *d_modularityGain;

    hipError_t err = hipMalloc(&d_edges, numEdges * sizeof(Edge));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for edges" << std::endl;
        return -1;
    }

    err = hipMalloc(&d_degrees, NUM_NODES * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for degrees" << std::endl;
        hipFree(d_edges);
        return -1;
    }

    err = hipMalloc(&d_communities, NUM_NODES * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for communities" << std::endl;
        hipFree(d_edges);
        hipFree(d_degrees);
        return -1;
    }

    err = hipMalloc(&d_modularityGain, NUM_NODES * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for modularity gain" << std::endl;
        hipFree(d_edges);
        hipFree(d_degrees);
        hipFree(d_communities);
        return -1;
    }

    // Copy edges to device
    err = hipMemcpy(d_edges, h_edges, numEdges * sizeof(Edge), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy edges to device" << std::endl;
        hipFree(d_edges);
        hipFree(d_degrees);
        hipFree(d_communities);
        hipFree(d_modularityGain);
        return -1;
    }

    // Initialize degrees array on device (for simplicity, set all to 1)
    thrust::device_ptr<int> d_degrees_ptr(d_degrees);
    thrust::fill(d_degrees_ptr, d_degrees_ptr + NUM_NODES, 1);

    // Initialize communities on device
    initializeCommunities<<<(NUM_NODES + 255) / 256, 256>>>(d_communities);
    hipDeviceSynchronize();

    // Initialize modularity gain on device
    thrust::device_ptr<float> d_modularityGain_ptr(d_modularityGain);
    thrust::fill(d_modularityGain_ptr, d_modularityGain_ptr + NUM_NODES, 0.0f);

    // Compute modularity gain
    computeModularityGain<<<(numEdges + 255) / 256, 256>>>(d_edges, d_degrees, d_communities, d_modularityGain, numEdges);
    hipDeviceSynchronize();

    // Check for errors
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel execution failed: " << hipGetErrorString(err) << std::endl;
        hipFree(d_edges);
        hipFree(d_degrees);
        hipFree(d_communities);
        hipFree(d_modularityGain);
        return -1;
    }

    // Free device memory
    hipFree(d_edges);
    hipFree(d_degrees);
    hipFree(d_communities);
    hipFree(d_modularityGain);

    std::cout << "Louvain community detection kernel executed successfully." << std::endl;

    return 0;
}