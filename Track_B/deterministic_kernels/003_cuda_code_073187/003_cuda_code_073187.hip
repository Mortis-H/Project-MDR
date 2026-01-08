#include <iostream>
#include <hip/hip_runtime.h>


#define NUM_NODES 128
#define BLOCK_SIZE 32

__device__ int community[NUM_NODES];
__device__ int node_degree[NUM_NODES];
__device__ int adjacency_matrix[NUM_NODES * NUM_NODES];

__global__ void initialize_communities() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_NODES) {
        community[idx] = idx; // Each node starts in its own community
    }
}

__global__ void compute_degrees() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_NODES) {
        int degree = 0;
        for (int j = 0; j < NUM_NODES; j++) {
            degree += adjacency_matrix[idx * NUM_NODES + j];
        }
        node_degree[idx] = degree;
    }
}

__global__ void update_communities() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_NODES) {
        int current_community = community[idx];
        int best_community = current_community;
        double best_modularity_gain = 0.0;

        for (int j = 0; j < NUM_NODES; j++) {
            if (adjacency_matrix[idx * NUM_NODES + j] > 0) {
                int neighbor_community = community[j];
                if (neighbor_community != current_community) {
                    // Simple modularity gain calculation
                    double delta_modularity = (2.0 * adjacency_matrix[idx * NUM_NODES + j] / NUM_NODES) -
                                              (static_cast<double>(node_degree[idx] * node_degree[j]) / (NUM_NODES * NUM_NODES));
                    if (delta_modularity > best_modularity_gain) {
                        best_modularity_gain = delta_modularity;
                        best_community = neighbor_community;
                    }
                }
            }
        }
        community[idx] = best_community;
    }
}

int main() {
    // Initialize host adjacency matrix
    int h_adjacency_matrix[NUM_NODES * NUM_NODES] = {0}; // Example: all zeros
    // For demonstration, let's create a simple ring graph
    for (int i = 0; i < NUM_NODES; i++) {
        h_adjacency_matrix[i * NUM_NODES + (i + 1) % NUM_NODES] = 1;
        h_adjacency_matrix[(i + 1) % NUM_NODES * NUM_NODES + i] = 1;
    }

    // Allocate device memory
    int *d_adjacency_matrix;
    hipError_t err = hipMalloc((void**)&d_adjacency_matrix, NUM_NODES * NUM_NODES * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for adjacency matrix" << std::endl;
        return -1;
    }

    // Copy adjacency matrix from host to device
    err = hipMemcpy(d_adjacency_matrix, h_adjacency_matrix, NUM_NODES * NUM_NODES * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy adjacency matrix from host to device" << std::endl;
        hipFree(d_adjacency_matrix);
        return -1;
    }

    // Initialize device pointers
    hipMemcpyToSymbol(HIP_SYMBOL(adjacency_matrix), &d_adjacency_matrix, sizeof(int*));

    // Initialize communities
    initialize_communities<<<(NUM_NODES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>();
    hipDeviceSynchronize();

    // Compute degrees
    compute_degrees<<<(NUM_NODES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>();
    hipDeviceSynchronize();

    // Update communities
    for (int iter = 0; iter < 10; iter++) { // Run for a fixed number of iterations
        update_communities<<<(NUM_NODES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>();
        hipDeviceSynchronize();
    }

    // Copy results back to host
    int h_community[NUM_NODES];
    hipMemcpyFromSymbol(h_community, HIP_SYMBOL(community), NUM_NODES * sizeof(int));

    // Print communities
    std::cout << "Community assignments:" << std::endl;
    for (int i = 0; i < NUM_NODES; i++) {
        std::cout << "Node " << i << ": Community " << h_community[i] << std::endl;
    }

    // Free device memory
    hipFree(d_adjacency_matrix);

    return 0;
}