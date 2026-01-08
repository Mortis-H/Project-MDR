#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>

namespace cg = cooperative_groups;

#define BLOCK_SIZE 256

__global__ void initialize_frontier(int *frontier, int *frontier_size, int start_node) {
    if (threadIdx.x == 0) {
        frontier[0] = start_node;
        *frontier_size = 1;
    }
}

__global__ void bfs_kernel(int *row_ptr, int *col_ind, int *frontier, int *new_frontier, int *visited, int *frontier_size, int *new_frontier_size, int num_nodes) {
    extern __shared__ int shared_frontier[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int size = *frontier_size;

    // Load frontier into shared memory
    if (gid < size) {
        shared_frontier[tid] = frontier[gid];
    }
    __syncthreads();

    if (gid < size) {
        int node = shared_frontier[tid];
        int start = row_ptr[node];
        int end = row_ptr[node + 1];
        for (int neighbor_idx = start; neighbor_idx < end; neighbor_idx++) {
            int neighbor = col_ind[neighbor_idx];
            int old_val = atomicExch(&visited[neighbor], 1);
            if (old_val == 0) {
                int idx = atomicAdd(new_frontier_size, 1);
                new_frontier[idx] = neighbor;
            }
        }
    }
}

void bfs(int *row_ptr, int *col_ind, int num_nodes, int start_node) {
    int *d_row_ptr, *d_col_ind, *d_frontier, *d_new_frontier, *d_visited, *d_frontier_size, *d_new_frontier_size;

    // Allocate device memory
    hipMalloc(&d_row_ptr, (num_nodes + 1) * sizeof(int));
    hipMalloc(&d_col_ind, row_ptr[num_nodes] * sizeof(int));
    hipMalloc(&d_frontier, num_nodes * sizeof(int));
    hipMalloc(&d_new_frontier, num_nodes * sizeof(int));
    hipMalloc(&d_visited, num_nodes * sizeof(int));
    hipMalloc(&d_frontier_size, sizeof(int));
    hipMalloc(&d_new_frontier_size, sizeof(int));

    // Initialize device memory
    hipMemcpy(d_row_ptr, row_ptr, (num_nodes + 1) * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_col_ind, col_ind, row_ptr[num_nodes] * sizeof(int), hipMemcpyHostToDevice);
    thrust::device_vector<int> d_visited_thrust(num_nodes);
    thrust::fill(d_visited_thrust.begin(), d_visited_thrust.end(), 0);
    hipMemcpy(d_visited, thrust::raw_pointer_cast(d_visited_thrust.data()), num_nodes * sizeof(int), hipMemcpyHostToDevice);

    // Initialize frontier with the start node
    initialize_frontier<<<1, 1>>>(d_frontier, d_frontier_size, start_node);
    hipDeviceSynchronize();

    int frontier_size = 1;
    while (frontier_size > 0) {
        // Reset new frontier size
        int zero = 0;
        hipMemcpy(d_new_frontier_size, &zero, sizeof(int), hipMemcpyHostToDevice);

        // Launch BFS kernel
        bfs_kernel<<<(frontier_size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(
            d_row_ptr, d_col_ind, d_frontier, d_new_frontier, d_visited, d_frontier_size, d_new_frontier_size, num_nodes);
        hipDeviceSynchronize();

        // Swap frontiers
        int *temp_frontier = d_frontier;
        d_frontier = d_new_frontier;
        d_new_frontier = temp_frontier;
        hipMemcpy(&frontier_size, d_frontier_size, sizeof(int), hipMemcpyDeviceToHost);
        hipMemcpy(d_frontier_size, d_new_frontier_size, sizeof(int), hipMemcpyDeviceToDevice);
    }

    // Free device memory
    hipFree(d_row_ptr);
    hipFree(d_col_ind);
    hipFree(d_frontier);
    hipFree(d_new_frontier);
    hipFree(d_visited);
    hipFree(d_frontier_size);
    hipFree(d_new_frontier_size);
}

int main() {
    // Example graph in COO format
    int num_nodes = 256;
    int num_edges = 512; // Example number of edges
    int row_ptr[257]; // Example row pointer array
    int col_ind[512]; // Example column index array

    // Initialize row_ptr and col_ind with example data
    // For simplicity, creating a ring graph
    for (int i = 0; i < num_nodes; i++) {
        row_ptr[i] = 2 * i;
        col_ind[2 * i] = (i + 1) % num_nodes;
        col_ind[2 * i + 1] = (i - 1 + num_nodes) % num_nodes;
    }
    row_ptr[num_nodes] = 2 * num_nodes;

    // Perform BFS starting from node 0
    bfs(row_ptr, col_ind, num_nodes, 0);

    // Check for errors
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    std::cout << "BFS completed successfully." << std::endl;
    return 0;
}