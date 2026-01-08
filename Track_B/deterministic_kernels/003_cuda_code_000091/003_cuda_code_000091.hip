#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <hiprand/hiprand_kernel.h>

namespace cg = cooperative_groups;

#define NUM_NODES 8192
#define MAX_ITERATIONS 100
#define BLOCK_SIZE 256

__global__ void init_labels(hiprandState *states, int *labels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_NODES) {
        hiprandState localState = states[idx];
        labels[idx] = hiprand(&localState) % NUM_NODES;
        states[idx] = localState;
    }
}

__global__ void label_propagation(int *labels, const int *edges, const int *edge_offsets, int *new_labels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_NODES) return;

    cg::thread_block cta = cg::this_thread_block();
    int local_min_label = labels[idx];
    int neighbors_start = edge_offsets[idx];
    int neighbors_end = edge_offsets[idx + 1];

    for (int neighbor_idx = neighbors_start; neighbor_idx < neighbors_end; ++neighbor_idx) {
        int neighbor = edges[neighbor_idx];
        int neighbor_label = labels[neighbor];
        local_min_label = min(local_min_label, neighbor_label);
    }

    cg::sync(cta);
    new_labels[idx] = local_min_label;
}

__global__ void init_curand_states(hiprandState *states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_NODES) {
        hiprand_init(seed, idx, 0, &states[idx]);
    }
}

int main() {
    // Allocate and initialize random states
    hiprandState *d_states;
    hipMalloc(&d_states, NUM_NODES * sizeof(hiprandState));
    init_curand_states<<<(NUM_NODES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_states, 1234);

    // Allocate and initialize labels
    int *d_labels, *d_new_labels;
    hipMalloc(&d_labels, NUM_NODES * sizeof(int));
    hipMalloc(&d_new_labels, NUM_NODES * sizeof(int));
    init_labels<<<(NUM_NODES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_states, d_labels);

    // Simulate an adjacency list for a simple graph (for demonstration purposes)
    int *h_edges, *h_edge_offsets;
    h_edges = new int[NUM_NODES * 10]; // Assume each node has at most 10 edges
    h_edge_offsets = new int[NUM_NODES + 1];
    thrust::host_vector<int> edges(NUM_NODES * 10);
    thrust::host_vector<int> edge_offsets(NUM_NODES + 1);

    // Simple graph initialization: each node connects to the next 10 nodes (wrap around)
    for (int i = 0; i < NUM_NODES; ++i) {
        edge_offsets[i] = i * 10;
        for (int j = 0; j < 10; ++j) {
            edges[i * 10 + j] = (i + j + 1) % NUM_NODES;
        }
    }
    edge_offsets[NUM_NODES] = NUM_NODES * 10;

    // Copy graph data to device
    int *d_edges, *d_edge_offsets;
    hipMalloc(&d_edges, NUM_NODES * 10 * sizeof(int));
    hipMalloc(&d_edge_offsets, (NUM_NODES + 1) * sizeof(int));
    hipMemcpy(d_edges, edges.data(), NUM_NODES * 10 * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_edge_offsets, edge_offsets.data(), (NUM_NODES + 1) * sizeof(int), hipMemcpyHostToDevice);

    // Label propagation algorithm
    bool labels_changed = true;
    int iteration = 0;
    while (labels_changed && iteration < MAX_ITERATIONS) {
        label_propagation<<<(NUM_NODES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_labels, d_edges, d_edge_offsets, d_new_labels);

        // Check if labels have changed
        hipMemcpy(d_labels, d_new_labels, NUM_NODES * sizeof(int), hipMemcpyDeviceToDevice);
        iteration++;
    }

    // Free memory
    hipFree(d_states);
    hipFree(d_labels);
    hipFree(d_new_labels);
    hipFree(d_edges);
    hipFree(d_edge_offsets);
    delete[] h_edges;
    delete[] h_edge_offsets;

    std::cout << "Label propagation completed in " << iteration << " iterations." << std::endl;

    return 0;
}