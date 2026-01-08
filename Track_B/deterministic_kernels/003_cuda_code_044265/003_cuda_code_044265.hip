#include <iostream>
#include <vector>
#include <hip/hip_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/scan.h>

#define MAX_NODES 8192
#define MAX_EDGES 16384

__global__ void initialize_frontier(int *frontier, int *visited, int start_node) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid == 0) {
        frontier[0] = start_node;
        visited[start_node] = 1;
    }
}

__global__ void bfs_kernel(int *frontier, int *new_frontier, int *visited, int *edge_list, int *edge_offsets, int *frontier_size, int *new_frontier_size) {
    extern __shared__ int shared_frontier[];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int lane_id = tid % 32;
    int warp_id = tid / 32;

    if (tid < *frontier_size) {
        int node = frontier[tid];
        int start = edge_offsets[node];
        int end = edge_offsets[node + 1];
        for (int i = start + lane_id; i < end; i += 32) {
            int neighbor = edge_list[i];
            if (!visited[neighbor]) {
                int index = atomicAdd(new_frontier_size, 1);
                new_frontier[index] = neighbor;
                atomicExch(&visited[neighbor], 1);
            }
        }
    }
    __syncthreads();

    if (lane_id == 0) {
        int warp_size = *new_frontier_size;
        shared_frontier[warp_id] = warp_size;
    }
    __syncthreads();

    if (warp_id == 0) {
        int prefix_sum = 0;
        for (int i = 0; i <= warp_id; ++i) {
            prefix_sum += shared_frontier[i];
        }
        shared_frontier[warp_id] = prefix_sum;
    }
    __syncthreads();

    if (tid < *new_frontier_size) {
        int new_index = shared_frontier[warp_id] + lane_id;
        if (new_index < *new_frontier_size) {
            frontier[new_index] = new_frontier[new_index];
        }
    }
}

__global__ void reset_new_frontier(int *new_frontier, int *new_frontier_size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid == 0) {
        *new_frontier_size = 0;
    }
}

int main() {
    // Initialize host data
    thrust::host_vector<int> h_edge_list(MAX_EDGES);
    thrust::host_vector<int> h_edge_offsets(MAX_NODES + 1);
    thrust::host_vector<int> h_visited(MAX_NODES, 0);
    thrust::host_vector<int> h_frontier(MAX_NODES);
    thrust::host_vector<int> h_new_frontier(MAX_NODES);
    int h_frontier_size = 1;
    int h_new_frontier_size = 0;

    // Example graph: fully connected for simplicity
    for (int i = 0; i < MAX_NODES; ++i) {
        h_edge_offsets[i] = i * MAX_NODES;
        for (int j = 0; j < MAX_NODES; ++j) {
            h_edge_list[i * MAX_NODES + j] = j;
        }
    }
    h_edge_offsets[MAX_NODES] = MAX_NODES * MAX_NODES;

    // Copy data to device
    thrust::device_vector<int> d_edge_list = h_edge_list;
    thrust::device_vector<int> d_edge_offsets = h_edge_offsets;
    thrust::device_vector<int> d_visited = h_visited;
    thrust::device_vector<int> d_frontier = h_frontier;
    thrust::device_vector<int> d_new_frontier = h_new_frontier;
    int *d_frontier_size;
    int *d_new_frontier_size;
    hipMalloc(&d_frontier_size, sizeof(int));
    hipMalloc(&d_new_frontier_size, sizeof(int));
    hipMemcpy(d_frontier_size, &h_frontier_size, sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_new_frontier_size, &h_new_frontier_size, sizeof(int), hipMemcpyHostToDevice);

    // Initialize frontier with starting node 0
    initialize_frontier<<<1, 1>>>(thrust::raw_pointer_cast(d_frontier.data()), thrust::raw_pointer_cast(d_visited.data()), 0);
    hipDeviceSynchronize();

    // BFS loop
    int num_iterations = 0;
    while (h_frontier_size > 0) {
        bfs_kernel<<<(h_frontier_size + 255) / 256, 256, 256 * sizeof(int)>>>(
            thrust::raw_pointer_cast(d_frontier.data()),
            thrust::raw_pointer_cast(d_new_frontier.data()),
            thrust::raw_pointer_cast(d_visited.data()),
            thrust::raw_pointer_cast(d_edge_list.data()),
            thrust::raw_pointer_cast(d_edge_offsets.data()),
            d_frontier_size,
            d_new_frontier_size
        );
        hipDeviceSynchronize();

        // Copy frontier sizes back to host
        hipMemcpy(&h_frontier_size, d_new_frontier_size, sizeof(int), hipMemcpyDeviceToHost);
        hipMemcpy(&h_new_frontier_size, d_new_frontier_size, sizeof(int), hipMemcpyDeviceToHost);

        // Swap frontiers
        thrust::device_vector<int> temp = d_frontier;
        d_frontier = d_new_frontier;
        d_new_frontier = temp;

        // Reset new frontier size
        reset_new_frontier<<<1, 1>>>(thrust::raw_pointer_cast(d_new_frontier.data()), d_new_frontier_size);
        hipDeviceSynchronize();

        num_iterations++;
    }

    // Copy visited array back to host
    thrust::copy(d_visited.begin(), d_visited.end(), h_visited.begin());

    // Output results
    std::cout << "BFS completed in " << num_iterations << " iterations." << std::endl;
    std::cout << "Visited nodes: ";
    for (int i = 0; i < MAX_NODES; ++i) {
        if (h_visited[i]) {
            std::cout << i << " ";
        }
    }
    std::cout << std::endl;

    // Cleanup
    hipFree(d_frontier_size);
    hipFree(d_new_frontier_size);

    return 0;
}