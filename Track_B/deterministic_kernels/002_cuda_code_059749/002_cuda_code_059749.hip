#include <iostream>
#include <vector>
#include <hip/hip_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>

#define BLOCK_SIZE 256

// Kernel to perform a single step of BFS
__global__ void bfsKernel(const int* row_ptr, const int* col_idx, int* distances, int* frontier, int* new_frontier, int* new_frontier_size, int level) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < frontier[level]) {
        int node = new_frontier[idx];
        int start = row_ptr[node];
        int end = row_ptr[node + 1];
        for (int neighbor = start; neighbor < end; ++neighbor) {
            int neighbor_id = col_idx[neighbor];
            if (atomicCAS(&distances[neighbor_id], -1, level + 1) == -1) {
                int pos = atomicAdd(new_frontier_size, 1);
                new_frontier[pos] = neighbor_id;
            }
        }
    }
}

// Kernel to clear the new frontier size
__global__ void clearNewFrontierSize(int* new_frontier_size) {
    *new_frontier_size = 0;
}

// Function to check CUDA errors
void checkCudaError(hipError_t err, const char* msg) {
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << msg << " - " << hipGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Example graph with 1024 nodes (randomly generated adjacency list)
    int num_nodes = 1024;
    std::vector<int> row_ptr(num_nodes + 1);
    std::vector<int> col_idx;

    // Generate a simple adjacency list for demonstration (each node connects to the next 4 nodes)
    row_ptr[0] = 0;
    for (int i = 0; i < num_nodes; ++i) {
        for (int j = 1; j <= 4; ++j) {
            if (i + j < num_nodes) {
                col_idx.push_back(i + j);
            }
        }
        row_ptr[i + 1] = col_idx.size();
    }

    // Initialize distances and frontier
    std::vector<int> distances(num_nodes, -1);
    std::vector<int> frontier(num_nodes);
    std::vector<int> new_frontier(num_nodes);
    int frontier_size = 1;
    distances[0] = 0;
    frontier[0] = 0;

    // Allocate device memory
    int* d_row_ptr, *d_col_idx, *d_distances, *d_frontier, *d_new_frontier, *d_new_frontier_size;
    checkCudaError(hipMalloc(&d_row_ptr, (num_nodes + 1) * sizeof(int)), "Failed to allocate d_row_ptr");
    checkCudaError(hipMalloc(&d_col_idx, col_idx.size() * sizeof(int)), "Failed to allocate d_col_idx");
    checkCudaError(hipMalloc(&d_distances, num_nodes * sizeof(int)), "Failed to allocate d_distances");
    checkCudaError(hipMalloc(&d_frontier, num_nodes * sizeof(int)), "Failed to allocate d_frontier");
    checkCudaError(hipMalloc(&d_new_frontier, num_nodes * sizeof(int)), "Failed to allocate d_new_frontier");
    checkCudaError(hipMalloc(&d_new_frontier_size, sizeof(int)), "Failed to allocate d_new_frontier_size");

    // Copy data to device
    checkCudaError(hipMemcpy(d_row_ptr, row_ptr.data(), (num_nodes + 1) * sizeof(int), hipMemcpyHostToDevice), "Failed to copy row_ptr to device");
    checkCudaError(hipMemcpy(d_col_idx, col_idx.data(), col_idx.size() * sizeof(int), hipMemcpyHostToDevice), "Failed to copy col_idx to device");
    checkCudaError(hipMemcpy(d_distances, distances.data(), num_nodes * sizeof(int), hipMemcpyHostToDevice), "Failed to copy distances to device");
    checkCudaError(hipMemcpy(d_frontier, frontier.data(), num_nodes * sizeof(int), hipMemcpyHostToDevice), "Failed to copy frontier to device");

    // Initialize new frontier size to 0
    clearNewFrontierSize<<<1, 1>>>(d_new_frontier_size);
    checkCudaError(hipGetLastError(), "Kernel launch failed");
    checkCudaError(hipDeviceSynchronize(), "Failed to synchronize");

    // Perform BFS
    int level = 0;
    while (frontier_size > 0) {
        bfsKernel<<<(frontier_size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_row_ptr, d_col_idx, d_distances, d_frontier, d_new_frontier, d_new_frontier_size, level);
        checkCudaError(hipGetLastError(), "Kernel launch failed");
        checkCudaError(hipDeviceSynchronize(), "Failed to synchronize");

        // Copy new frontier size to host
        int h_new_frontier_size;
        checkCudaError(hipMemcpy(&h_new_frontier_size, d_new_frontier_size, sizeof(int), hipMemcpyDeviceToHost), "Failed to copy new_frontier_size to host");

        // Swap frontiers and clear new frontier size
        std::swap(frontier, new_frontier);
        frontier_size = h_new_frontier_size;
        clearNewFrontierSize<<<1, 1>>>(d_new_frontier_size);
        checkCudaError(hipGetLastError(), "Kernel launch failed");
        checkCudaError(hipDeviceSynchronize(), "Failed to synchronize");

        level++;
    }

    // Copy distances back to host
    checkCudaError(hipMemcpy(distances.data(), d_distances, num_nodes * sizeof(int), hipMemcpyDeviceToHost), "Failed to copy distances back to host");

    // Print distances (for verification)
    for (int i = 0; i < 10; ++i) {
        std::cout << "Node " << i << ": Distance " << distances[i] << std::endl;
    }

    // Free device memory
    hipFree(d_row_ptr);
    hipFree(d_col_idx);
    hipFree(d_distances);
    hipFree(d_frontier);
    hipFree(d_new_frontier);
    hipFree(d_new_frontier_size);

    return 0;
}