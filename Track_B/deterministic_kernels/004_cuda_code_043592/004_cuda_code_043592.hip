#include <iostream>
#include <hip/hip_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/equal.h>
#include <thrust/unique.h>
#include <thrust/sequence.h>
#include <thrust/extrema.h>
#include <thrust/gather.h>
#include <thrust/remove.h>
#include <thrust/functional.h>

#define BLOCK_SIZE 256
#define NUM_NODES 16384

__global__ void initialize_labels(int *labels, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        labels[idx] = idx;
    }
}

__global__ void propagate_labels(int *labels, const int *edges, const int *edge_offsets, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int start = edge_offsets[idx];
        int end = edge_offsets[idx + 1];
        int min_label = labels[idx];
        for (int j = start; j < end; ++j) {
            min_label = min(min_label, labels[edges[j]]);
        }
        labels[idx] = min_label;
    }
}

__global__ void compress_labels(int *labels, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        labels[idx] = labels[labels[idx]];
    }
}

__global__ void build_reverse_edges(const int *edges, const int *edge_offsets, int *reverse_edges, int *reverse_edge_offsets, int n, int num_edges) {
    extern __shared__ int shared_indices[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int start = edge_offsets[idx];
    int end = edge_offsets[idx + 1];
    int count = 0;
    for (int j = start; j < end; ++j) {
        int neighbor = edges[j];
        int pos = atomicAdd(&reverse_edge_offsets[neighbor + 1], 1);
        shared_indices[pos] = idx;
    }
    __syncthreads();
    start = reverse_edge_offsets[idx];
    end = reverse_edge_offsets[idx + 1];
    for (int j = start; j < end; ++j) {
        reverse_edges[j] = shared_indices[j - start];
    }
}

int main() {
    // Example graph data
    thrust::host_vector<int> h_edges = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}; // Example edges
    thrust::host_vector<int> h_edge_offsets(NUM_NODES + 1, 0); // Example edge offsets
    for (int i = 0; i < NUM_NODES; ++i) {
        h_edge_offsets[i + 1] = h_edge_offsets[i] + 1; // Example: each node has one outgoing edge
    }

    thrust::device_vector<int> d_edges = h_edges;
    thrust::device_vector<int> d_edge_offsets = h_edge_offsets;
    thrust::device_vector<int> d_labels(NUM_NODES);
    thrust::device_vector<int> d_reverse_edges(h_edges.size());
    thrust::device_vector<int> d_reverse_edge_offsets(NUM_NODES + 1, 0);

    // Initialize labels
    initialize_labels<<<(NUM_NODES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(thrust::raw_pointer_cast(d_labels.data()), NUM_NODES);
    hipDeviceSynchronize();

    // Compute strongly connected components
    bool changed = true;
    while (changed) {
        changed = false;

        // Propagate labels
        propagate_labels<<<(NUM_NODES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(thrust::raw_pointer_cast(d_labels.data()), thrust::raw_pointer_cast(d_edges.data()), thrust::raw_pointer_cast(d_edge_offsets.data()), NUM_NODES);
        hipDeviceSynchronize();

        // Compress labels
        compress_labels<<<(NUM_NODES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(thrust::raw_pointer_cast(d_labels.data()), NUM_NODES);
        hipDeviceSynchronize();

        // Check for changes
        thrust::device_vector<int> d_old_labels = d_labels;
        compress_labels<<<(NUM_NODES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(thrust::raw_pointer_cast(d_labels.data()), NUM_NODES);
        hipDeviceSynchronize();
        changed = !thrust::equal(d_old_labels.begin(), d_old_labels.end(), d_labels.begin());
    }

    // Build reverse edges
    int num_edges = h_edges.size();
    build_reverse_edges<<<(NUM_NODES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(thrust::raw_pointer_cast(d_edges.data()), thrust::raw_pointer_cast(d_edge_offsets.data()), thrust::raw_pointer_cast(d_reverse_edges.data()), thrust::raw_pointer_cast(d_reverse_edge_offsets.data()), NUM_NODES, num_edges);
    hipDeviceSynchronize();

    // Scan to get correct offsets
    thrust::exclusive_scan(d_reverse_edge_offsets.begin(), d_reverse_edge_offsets.end(), d_reverse_edge_offsets.begin());

    // Propagate labels on reverse graph
    changed = true;
    while (changed) {
        changed = false;

        // Propagate labels
        propagate_labels<<<(NUM_NODES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(thrust::raw_pointer_cast(d_labels.data()), thrust::raw_pointer_cast(d_reverse_edges.data()), thrust::raw_pointer_cast(d_reverse_edge_offsets.data()), NUM_NODES);
        hipDeviceSynchronize();

        // Compress labels
        compress_labels<<<(NUM_NODES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(thrust::raw_pointer_cast(d_labels.data()), NUM_NODES);
        hipDeviceSynchronize();

        // Check for changes
        thrust::device_vector<int> d_old_labels = d_labels;
        compress_labels<<<(NUM_NODES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(thrust::raw_pointer_cast(d_labels.data()), NUM_NODES);
        hipDeviceSynchronize();
        changed = !thrust::equal(d_old_labels.begin(), d_old_labels.end(), d_labels.begin());
    }

    // Copy results back to host
    thrust::host_vector<int> h_labels = d_labels;

    // Print results
    std::cout << "Strongly Connected Components:\n";
    for (int i = 0; i < NUM_NODES; ++i) {
        std::cout << "Node " << i << " -> Component " << h_labels[i] << "\n";
    }

    return 0;
}