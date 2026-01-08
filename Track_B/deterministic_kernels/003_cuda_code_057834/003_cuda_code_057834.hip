#include <iostream>
#include <hip/hip_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/gather.h>
#include <thrust/remove.h>
#include <thrust/execution_policy.h>

#define BLOCK_SIZE 256
#define NUM_NODES 512

__global__ void propagate_labels(int* labels, const int* edges, const int* edge_counts, int* active_nodes, int num_active) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_active) return;

    int node = active_nodes[idx];
    int start = edge_counts[node];
    int end = edge_counts[node + 1];
    int label = labels[node];

    for (int i = start; i < end; ++i) {
        int neighbor = edges[i];
        atomicMin(&labels[neighbor], label);
    }
}

__global__ void identify_active_nodes(int* labels, int* active_nodes, int* num_active) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_NODES) return;

    int label = labels[idx];
    if (label != idx) {
        int pos = atomicAdd(num_active, 1);
        active_nodes[pos] = idx;
    }
}

__global__ void initialize_labels(int* labels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_NODES) {
        labels[idx] = idx;
    }
}

int main() {
    // Example graph edges (adjacency list representation)
    // Each node i has edges to edges[edge_counts[i]:edge_counts[i+1]]
    int h_edges[] = {1, 2, 0, 3, 0, 4, 1, 2, 2, 3, 3, 4, 4, 0, 4, 1};
    int h_edge_counts[] = {0, 2, 4, 6, 8, 10, 12, 14, 16};

    thrust::device_vector<int> d_edges(h_edges, h_edges + 16);
    thrust::device_vector<int> d_edge_counts(h_edge_counts, h_edge_counts + 8);

    thrust::device_vector<int> d_labels(NUM_NODES);
    thrust::device_vector<int> d_active_nodes(NUM_NODES);
    thrust::device_vector<int> d_num_active(1, 0);

    initialize_labels<<<(NUM_NODES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(thrust::raw_pointer_cast(d_labels.data()));

    bool converged = false;
    while (!converged) {
        thrust::fill(thrust::device, d_num_active.begin(), d_num_active.end(), 0);
        identify_active_nodes<<<(NUM_NODES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(thrust::raw_pointer_cast(d_labels.data()),
                                                                                     thrust::raw_pointer_cast(d_active_nodes.data()),
                                                                                     thrust::raw_pointer_cast(d_num_active.data()));

        int num_active = 0;
        hipMemcpy(&num_active, thrust::raw_pointer_cast(d_num_active.data()), sizeof(int), hipMemcpyDeviceToHost);

        if (num_active == 0) {
            converged = true;
        } else {
            propagate_labels<<<(num_active + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(thrust::raw_pointer_cast(d_labels.data()),
                                                                                       thrust::raw_pointer_cast(d_edges.data()),
                                                                                       thrust::raw_pointer_cast(d_edge_counts.data()),
                                                                                       thrust::raw_pointer_cast(d_active_nodes.data()),
                                                                                       num_active);
        }
    }

    // Copy results back to host for verification
    thrust::host_vector<int> h_labels = d_labels;

    // Output the labels to verify correctness
    std::cout << "Strongly Connected Components Labels:" << std::endl;
    for (int i = 0; i < NUM_NODES; ++i) {
        std::cout << "Node " << i << ": Component " << h_labels[i] << std::endl;
    }

    // Check for any CUDA errors
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return 1;
    }

    return 0;
}