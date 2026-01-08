#include <iostream>
#include <hip/hip_runtime.h>

#include <limits>

#define NUM_PARTICLES 8192
#define WARP_SIZE 32

__device__ int findMinVertex(int* distances, bool* visited, int start, int end) {
    int minDist = INT_MAX;
    int minVertex = -1;
    for (int i = start; i < end; ++i) {
        if (!visited[i] && distances[i] < minDist) {
            minDist = distances[i];
            minVertex = i;
        }
    }
    return minVertex;
}

__device__ void updateDistances(int* distances, bool* visited, int u, int* graph, int numParticles) {
    int tid = threadIdx.x;
    int stride = blockDim.x;
    for (int v = tid; v < numParticles; v += stride) {
        if (!visited[v] && graph[u * numParticles + v] && distances[u] != INT_MAX) {
            int newDist = distances[u] + graph[u * numParticles + v];
            atomicMin(&distances[v], newDist);
        }
    }
    __syncthreads();
}

__global__ void dijkstraKernel(int* distances, bool* visited, int* graph, int numParticles) {
    extern __shared__ int s_distances[];
    extern __shared__ bool s_visited[];

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    if (tid == 0) {
        distances[bid] = 0;
    }
    __syncthreads();

    for (int i = 0; i < numParticles - 1; ++i) {
        int u = findMinVertex(distances, visited, bid * WARP_SIZE, (bid + 1) * WARP_SIZE);
        if (u != -1) {
            visited[u] = true;
            updateDistances(distances, visited, u, graph, numParticles);
        }
    }
}

__global__ void initialize(int* distances, bool* visited, int numParticles) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < numParticles) {
        distances[tid] = INT_MAX;
        visited[tid] = false;
    }
}

int main() {
    int* h_graph;
    int* h_distances;
    bool* h_visited;

    int* d_graph;
    int* d_distances;
    bool* d_visited;

    h_graph = new int[NUM_PARTICLES * NUM_PARTICLES];
    h_distances = new int[NUM_PARTICLES];
    h_visited = new bool[NUM_PARTICLES];

    hipError_t err;

    err = hipMalloc((void**)&d_graph, NUM_PARTICLES * NUM_PARTICLES * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Error allocating d_graph: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    err = hipMalloc((void**)&d_distances, NUM_PARTICLES * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Error allocating d_distances: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    err = hipMalloc((void**)&d_visited, NUM_PARTICLES * sizeof(bool));
    if (err != hipSuccess) {
        std::cerr << "Error allocating d_visited: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Initialize graph (example: fully connected graph with random weights)
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        for (int j = 0; j < NUM_PARTICLES; ++j) {
            h_graph[i * NUM_PARTICLES + j] = (i != j) ? (rand() % 100 + 1) : 0;
        }
    }

    err = hipMemcpy(d_graph, h_graph, NUM_PARTICLES * NUM_PARTICLES * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Error copying graph to device: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    int blockSize = WARP_SIZE;
    int numBlocks = NUM_PARTICLES / WARP_SIZE;

    initialize<<<numBlocks, blockSize>>>(d_distances, d_visited, NUM_PARTICLES);
    hipDeviceSynchronize();

    size_t sharedMemSize = WARP_SIZE * (sizeof(int) + sizeof(bool));
    dijkstraKernel<<<numBlocks, blockSize, sharedMemSize>>>(d_distances, d_visited, d_graph, NUM_PARTICLES);
    hipDeviceSynchronize();

    err = hipMemcpy(h_distances, d_distances, NUM_PARTICLES * sizeof(int), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Error copying distances back to host: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Output distances from the source node (0) to all other nodes
    std::cout << "Distance from source node 0 to all other nodes:\n";
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        std::cout << "Node " << i << ": " << h_distances[i] << "\n";
    }

    hipFree(d_graph);
    hipFree(d_distances);
    hipFree(d_visited);

    delete[] h_graph;
    delete[] h_distances;
    delete[] h_visited;

    return 0;
}