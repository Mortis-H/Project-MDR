#include <iostream>
#include <hip/hip_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/unique.h>

// Kernel to perform the first BFS pass
__global__ void bfs_kernel(int *d_graph, int *d_distances, int *d_visited, int num_nodes, int source) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;

    int start = idx * num_nodes;
    int distance = d_distances[idx];
    if (distance == -1) return;

    for (int i = 0; i < num_nodes; ++i) {
        int neighbor = d_graph[start + i];
        if (neighbor == 1 && atomicCAS(&d_visited[i], 0, 1) == 0) {
            atomicMin(&d_distances[i], distance + 1);
        }
    }
}

// Kernel to perform the second BFS pass in reverse
__global__ void reverse_bfs_kernel(int *d_graph, int *d_distances, int *d_visited, int num_nodes, int source) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;

    int start = idx * num_nodes;
    int distance = d_distances[idx];
    if (distance == -1) return;

    for (int i = 0; i < num_nodes; ++i) {
        int neighbor = d_graph[start + i];
        if (neighbor == 1 && atomicCAS(&d_visited[i], 0, 1) == 0) {
            atomicMin(&d_distances[i], distance + 1);
        }
    }
}

// Kernel to assign SCC labels
__global__ void assign_scc_labels(int *d_distances, int *d_scc_labels, int *d_scc_count, int num_nodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;

    int distance = d_distances[idx];
    if (distance == -1) return;

    int label = atomicCAS(&d_scc_labels[idx], -1, distance);
    if (label == -1) {
        atomicAdd(d_scc_count, 1);
    }
}

int main() {
    const int num_nodes = 512;
    const int num_edges = 1024; // Example number of edges

    // Host graph adjacency matrix
    thrust::host_vector<int> h_graph(num_nodes * num_nodes, 0);
    // Randomly generate edges for demonstration
    for (int i = 0; i < num_edges; ++i) {
        int src = rand() % num_nodes;
        int dst = rand() % num_nodes;
        h_graph[src * num_nodes + dst] = 1;
    }

    // Device graph adjacency matrix
    thrust::device_vector<int> d_graph(h_graph);
    thrust::device_vector<int> d_distances(num_nodes, -1);
    thrust::device_vector<int> d_visited(num_nodes, 0);
    thrust::device_vector<int> d_scc_labels(num_nodes, -1);
    thrust::device_vector<int> d_scc_count(1, 0);

    // Set source node for BFS
    int source = 0;
    d_distances[source] = 0;

    // Number of threads and blocks
    const int threads_per_block = 256;
    const int num_blocks = (num_nodes + threads_per_block - 1) / threads_per_block;

    // Perform first BFS
    bfs_kernel<<<num_blocks, threads_per_block>>>(thrust::raw_pointer_cast(d_graph.data()),
                                                 thrust::raw_pointer_cast(d_distances.data()),
                                                 thrust::raw_pointer_cast(d_visited.data()),
                                                 num_nodes, source);
    hipDeviceSynchronize();

    // Reset visited array
    thrust::fill(d_visited.begin(), d_visited.end(), 0);

    // Perform second BFS in reverse
    reverse_bfs_kernel<<<num_blocks, threads_per_block>>>(thrust::raw_pointer_cast(d_graph.data()),
                                                         thrust::raw_pointer_cast(d_distances.data()),
                                                         thrust::raw_pointer_cast(d_visited.data()),
                                                         num_nodes, source);
    hipDeviceSynchronize();

    // Assign SCC labels
    assign_scc_labels<<<num_blocks, threads_per_block>>>(thrust::raw_pointer_cast(d_distances.data()),
                                                        thrust::raw_pointer_cast(d_scc_labels.data()),
                                                        thrust::raw_pointer_cast(d_scc_count.data()),
                                                        num_nodes);
    hipDeviceSynchronize();

    // Copy results back to host
    thrust::host_vector<int> h_scc_labels(d_scc_labels);
    thrust::host_vector<int> h_scc_count(d_scc_count);

    // Output the number of SCCs and labels
    std::cout << "Number of SCCs: " << h_scc_count[0] << std::endl;
    std::cout << "SCC Labels: ";
    for (int i = 0; i < num_nodes; ++i) {
        std::cout << h_scc_labels[i] << " ";
    }
    std::cout << std::endl;

    // Error checking
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    return 0;
}