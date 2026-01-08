#include <iostream>
#include <hip/hip_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/fill.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/gather.h>
#include <thrust/remove.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <hiprand/hiprand_kernel.h>

#define NUM_NODES 512
#define MAX_COLOR 16
#define BLOCK_SIZE 256

__global__ void initRandomEdges(int *edges, int *edgeCounts, int maxEdgesPerNode, hiprandState *states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_NODES) return;
    hiprandState localState = states[idx];
    int numEdges = hiprand(&localState) % maxEdgesPerNode;
    edgeCounts[idx] = numEdges;
    for (int i = 0; i < numEdges; ++i) {
        int target = hiprand(&localState) % NUM_NODES;
        while (target == idx) {
            target = hiprand(&localState) % NUM_NODES;
        }
        edges[idx * maxEdgesPerNode + i] = target;
    }
    states[idx] = localState;
}

__global__ void assignInitialColors(int *colors, int *edgeCounts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_NODES) return;
    colors[idx] = edgeCounts[idx] % MAX_COLOR;
}

__global__ void updateColors(int *colors, int *edges, int *edgeCounts, int maxEdgesPerNode) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_NODES) return;
    int currentColor = colors[idx];
    int newColor = currentColor;
    for (int i = 0; i < edgeCounts[idx]; ++i) {
        int neighbor = edges[idx * maxEdgesPerNode + i];
        int neighborColor = colors[neighbor];
        if (neighborColor == newColor) {
            newColor = (newColor + 1) % MAX_COLOR;
            i = -1; // Restart the loop to check all neighbors again
        }
    }
    colors[idx] = newColor;
}

__global__ void initCurandStates(hiprandState *states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_NODES) return;
    hiprand_init(seed, idx, 0, &states[idx]);
}

int main() {
    int maxEdgesPerNode = 10;
    int *h_edges = new int[NUM_NODES * maxEdgesPerNode];
    int *h_edgeCounts = new int[NUM_NODES];
    int *h_colors = new int[NUM_NODES];

    thrust::device_vector<int> d_edges(NUM_NODES * maxEdgesPerNode);
    thrust::device_vector<int> d_edgeCounts(NUM_NODES);
    thrust::device_vector<int> d_colors(NUM_NODES);
    thrust::device_vector<hiprandState> d_states(NUM_NODES);

    // Initialize random states
    initCurandStates<<<(NUM_NODES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(thrust::raw_pointer_cast(d_states.data()), 123456ULL);
    hipDeviceSynchronize();

    // Generate random edges
    initRandomEdges<<<(NUM_NODES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(thrust::raw_pointer_cast(d_edges.data()),
                                                                             thrust::raw_pointer_cast(d_edgeCounts.data()),
                                                                             maxEdgesPerNode,
                                                                             thrust::raw_pointer_cast(d_states.data()));
    hipDeviceSynchronize();

    // Assign initial colors
    assignInitialColors<<<(NUM_NODES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(thrust::raw_pointer_cast(d_colors.data()),
                                                                                 thrust::raw_pointer_cast(d_edgeCounts.data()));
    hipDeviceSynchronize();

    // Update colors iteratively
    for (int iter = 0; iter < 10; ++iter) { // Run for a fixed number of iterations
        updateColors<<<(NUM_NODES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(thrust::raw_pointer_cast(d_colors.data()),
                                                                               thrust::raw_pointer_cast(d_edges.data()),
                                                                               thrust::raw_pointer_cast(d_edgeCounts.data()),
                                                                               maxEdgesPerNode);
        hipDeviceSynchronize();
    }

    // Copy results back to host
    hipMemcpy(h_colors, thrust::raw_pointer_cast(d_colors.data()), NUM_NODES * sizeof(int), hipMemcpyDeviceToHost);

    // Print the colors
    std::cout << "Node colors:" << std::endl;
    for (int i = 0; i < NUM_NODES; ++i) {
        std::cout << "Node " << i << ": Color " << h_colors[i] << std::endl;
    }

    delete[] h_edges;
    delete[] h_edgeCounts;
    delete[] h_colors;

    return 0;
}