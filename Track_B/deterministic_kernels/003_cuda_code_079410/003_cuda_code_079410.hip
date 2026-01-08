#include <iostream>
#include <hip/hip_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/fill.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/equal.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/adjacent_difference.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/remove.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>

#define NUM_NODES 8192

__global__ void propagateLabels(int *labels, const int *edges, const int *edgeCounts, int *nextLabels, int numNodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numNodes) return;

    int label = labels[idx];
    int start = edgeCounts[idx];
    int end = edgeCounts[idx + 1];

    for (int j = start; j < end; ++j) {
        int neighbor = edges[j];
        atomicMin(&nextLabels[neighbor], label);
    }
}

__global__ void updateLabels(int *labels, const int *nextLabels, int numNodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numNodes) return;
    labels[idx] = nextLabels[idx];
}

__global__ void clearNextLabels(int *nextLabels, int numNodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numNodes) return;
    nextLabels[idx] = idx;
}

int main() {
    // Initialize graph edges and edge counts
    thrust::host_vector<int> h_edges; // Example edges, should be filled with actual graph data
    thrust::host_vector<int> h_edgeCounts(NUM_NODES + 1, 0); // Example edge counts, should be filled with actual graph data

    // Example graph: fully connected for simplicity
    for (int i = 0; i < NUM_NODES; ++i) {
        for (int j = 0; j < NUM_NODES; ++j) {
            h_edges.push_back(j);
        }
        h_edgeCounts[i + 1] = h_edgeCounts[i] + NUM_NODES;
    }

    // Copy data to device
    thrust::device_vector<int> d_edges = h_edges;
    thrust::device_vector<int> d_edgeCounts = h_edgeCounts;

    // Initialize labels
    thrust::device_vector<int> d_labels(NUM_NODES);
    thrust::sequence(d_labels.begin(), d_labels.end());

    // Allocate memory for next labels
    thrust::device_vector<int> d_nextLabels(NUM_NODES);
    thrust::sequence(d_nextLabels.begin(), d_nextLabels.end());

    // Kernel configuration
    int blockSize = 256;
    int numBlocks = (NUM_NODES + blockSize - 1) / blockSize;

    // Iteratively propagate labels
    bool changed = true;
    while (changed) {
        clearNextLabels<<<numBlocks, blockSize>>>(thrust::raw_pointer_cast(d_nextLabels.data()), NUM_NODES);
        hipError_t err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Error in clearNextLabels: " << hipGetErrorString(err) << std::endl;
            return -1;
        }

        propagateLabels<<<numBlocks, blockSize>>>(thrust::raw_pointer_cast(d_labels.data()),
                                                thrust::raw_pointer_cast(d_edges.data()),
                                                thrust::raw_pointer_cast(d_edgeCounts.data()),
                                                thrust::raw_pointer_cast(d_nextLabels.data()),
                                                NUM_NODES);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Error in propagateLabels: " << hipGetErrorString(err) << std::endl;
            return -1;
        }

        updateLabels<<<numBlocks, blockSize>>>(thrust::raw_pointer_cast(d_labels.data()),
                                             thrust::raw_pointer_cast(d_nextLabels.data()),
                                             NUM_NODES);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Error in updateLabels: " << hipGetErrorString(err) << std::endl;
            return -1;
        }

        // Check for changes
        thrust::host_vector<int> h_labels(d_labels);
        thrust::host_vector<int> h_nextLabels(d_nextLabels);
        changed = !thrust::equal(h_labels.begin(), h_labels.end(), h_nextLabels.begin());
    }

    // Output the result
    thrust::host_vector<int> h_finalLabels(d_labels);
    std::cout << "Strongly connected components labels:" << std::endl;
    for (int i = 0; i < NUM_NODES; ++i) {
        std::cout << "Node " << i << ": Component " << h_finalLabels[i] << std::endl;
    }

    return 0;
}