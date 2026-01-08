#include <iostream>
#include <hip/hip_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <hiprand/hiprand_kernel.h>

// Kernel to initialize random values
__global__ void init_random(hiprandState *state, int seed) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    hiprand_init(seed, id, 0, &state[id]);
}

// Kernel to perform a single iteration of PageRank
__global__ void pagerank_kernel(float *scores, float *new_scores, int *out_degrees, int *adjacency_list, int num_nodes, float damping_factor) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node < num_nodes) {
        float score = 0.0f;
        int start = node == 0 ? 0 : adjacency_list[node - 1];
        int end = adjacency_list[node];
        for (int neighbor = start; neighbor < end; ++neighbor) {
            int target = adjacency_list[neighbor];
            score += scores[target] / out_degrees[target];
        }
        new_scores[node] = (1.0f - damping_factor) / num_nodes + damping_factor * score;
    }
}

int main() {
    const int num_nodes = 512;
    const int num_iterations = 10;
    const float damping_factor = 0.85f;

    // Initialize host data
    thrust::host_vector<float> h_scores(num_nodes, 1.0f / num_nodes);
    thrust::host_vector<float> h_new_scores(num_nodes, 0.0f);
    thrust::host_vector<int> h_out_degrees(num_nodes, 1);
    thrust::host_vector<int> h_adjacency_list;

    // Randomly generate adjacency list for a simple graph
    int num_edges = num_nodes * 2; // Example: each node has 2 outgoing edges
    h_adjacency_list.resize(num_edges);
    hiprandState *dev_states;
    hipMalloc(&dev_states, num_nodes * sizeof(hiprandState));

    init_random<<<(num_nodes + 255) / 256, 256>>>(dev_states, 1234);
    hipDeviceSynchronize();

    // Simple random adjacency list generation
    for (int i = 0; i < num_edges; ++i) {
        h_adjacency_list[i] = i % num_nodes;
    }

    // Allocate device memory
    float *d_scores, *d_new_scores;
    int *d_out_degrees, *d_adjacency_list;
    hipMalloc(&d_scores, num_nodes * sizeof(float));
    hipMalloc(&d_new_scores, num_nodes * sizeof(float));
    hipMalloc(&d_out_degrees, num_nodes * sizeof(int));
    hipMalloc(&d_adjacency_list, num_edges * sizeof(int));

    // Copy data to device
    hipMemcpy(d_scores, h_scores.data(), num_nodes * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_new_scores, h_new_scores.data(), num_nodes * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_out_degrees, h_out_degrees.data(), num_nodes * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_adjacency_list, h_adjacency_list.data(), num_edges * sizeof(int), hipMemcpyHostToDevice);

    // Perform PageRank iterations
    for (int iter = 0; iter < num_iterations; ++iter) {
        pagerank_kernel<<<(num_nodes + 255) / 256, 255>>>(d_scores, d_new_scores, d_out_degrees, d_adjacency_list, num_nodes, damping_factor);
        hipDeviceSynchronize();
        std::swap(d_scores, d_new_scores);
    }

    // Copy results back to host
    hipMemcpy(h_scores.data(), d_scores, num_nodes * sizeof(float), hipMemcpyDeviceToHost);

    // Sort scores using Thrust
    thrust::device_vector<float> d_scores_thrust(h_scores);
    thrust::sort(thrust::device, d_scores_thrust.begin(), d_scores_thrust.end());

    // Print sorted scores
    std::cout << "Sorted PageRank scores:" << std::endl;
    for (int i = 0; i < num_nodes; ++i) {
        std::cout << d_scores_thrust[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    hipFree(d_scores);
    hipFree(d_new_scores);
    hipFree(d_out_degrees);
    hipFree(d_adjacency_list);
    hipFree(dev_states);

    return 0;
}