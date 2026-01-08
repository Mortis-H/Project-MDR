#include <iostream>
#include <hip/hip_runtime.h>


#define BLOCK_SIZE 256
#define MAX_EDGES 512

__global__ void initializeFrontier(int *frontier, int *newFrontier, int *visited, int *level, int startNode) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        frontier[idx] = startNode;
        newFrontier[idx] = -1;
        visited[startNode] = 1;
        level[startNode] = 0;
    } else if (idx < MAX_EDGES) {
        frontier[idx] = -1;
        newFrontier[idx] = -1;
        visited[idx] = 0;
        level[idx] = -1;
    }
}

__global__ void bfsKernel(int *frontier, int *newFrontier, int *visited, int *level, const int *adjMatrix, int numNodes, int *newFrontierSize) {
    extern __shared__ int sharedFrontier[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int sharedIdx = threadIdx.x;

    // Load frontier into shared memory
    if (idx < numNodes && frontier[idx] != -1) {
        sharedFrontier[sharedIdx] = frontier[idx];
    } else {
        sharedFrontier[sharedIdx] = -1;
    }
    __syncthreads();

    // Process each node in the frontier
    if (sharedFrontier[sharedIdx] != -1) {
        int currentNode = sharedFrontier[sharedIdx];
        int currentLevel = level[currentNode];
        for (int i = 0; i < numNodes; ++i) {
            int edge = adjMatrix[currentNode * numNodes + i];
            if (edge && !visited[i]) {
                atomicExch(&visited[i], 1);
                int pos = atomicAdd(newFrontierSize, 1);
                newFrontier[pos] = i;
                level[i] = currentLevel + 1;
            }
        }
    }
}

__global__ void clearNewFrontier(int *newFrontier, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        newFrontier[idx] = -1;
    }
}

void bfs(int *adjMatrix, int numNodes, int startNode) {
    int *d_frontier, *d_newFrontier, *d_visited, *d_level, *d_newFrontierSize;
    int h_newFrontierSize = 0;

    hipMalloc((void**)&d_frontier, numNodes * sizeof(int));
    hipMalloc((void**)&d_newFrontier, numNodes * sizeof(int));
    hipMalloc((void**)&d_visited, numNodes * sizeof(int));
    hipMalloc((void**)&d_level, numNodes * sizeof(int));
    hipMalloc((void**)&d_newFrontierSize, sizeof(int));

    // Initialize device arrays
    initializeFrontier<<<(numNodes + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_frontier, d_newFrontier, d_visited, d_level, startNode);
    hipDeviceSynchronize();

    // Perform BFS
    while (h_newFrontierSize < numNodes) {
        hipMemset(d_newFrontierSize, 0, sizeof(int));
        bfsKernel<<<(numNodes + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(d_frontier, d_newFrontier, d_visited, d_level, adjMatrix, numNodes, d_newFrontierSize);
        hipDeviceSynchronize();

        hipMemcpy(&h_newFrontierSize, d_newFrontierSize, sizeof(int), hipMemcpyDeviceToHost);
        if (h_newFrontierSize == 0) break;

        // Swap frontiers
        int *temp = d_frontier;
        d_frontier = d_newFrontier;
        d_newFrontier = temp;

        // Clear new frontier
        clearNewFrontier<<<(h_newFrontierSize + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_newFrontier, h_newFrontierSize);
        hipDeviceSynchronize();
    }

    // Copy results back to host for verification
    int *h_level = new int[numNodes];
    hipMemcpy(h_level, d_level, numNodes * sizeof(int), hipMemcpyDeviceToHost);

    // Print results
    for (int i = 0; i < numNodes; ++i) {
        std::cout << "Node " << i << " level: " << h_level[i] << std::endl;
    }

    // Free device memory
    hipFree(d_frontier);
    hipFree(d_newFrontier);
    hipFree(d_visited);
    hipFree(d_level);
    hipFree(d_newFrontierSize);
    delete[] h_level;
}

int main() {
    int numNodes = 512;
    int startNode = 0;

    // Allocate and initialize adjacency matrix on host
    int *adjMatrix = new int[numNodes * numNodes];
    for (int i = 0; i < numNodes * numNodes; ++i) {
        adjMatrix[i] = 0;
    }

    // Example: Create a simple ring graph
    for (int i = 0; i < numNodes; ++i) {
        adjMatrix[i * numNodes + (i + 1) % numNodes] = 1;
        adjMatrix[i * numNodes + (i - 1 + numNodes) % numNodes] = 1;
    }

    // Allocate adjacency matrix on device
    int *d_adjMatrix;
    hipMalloc((void**)&d_adjMatrix, numNodes * numNodes * sizeof(int));
    hipMemcpy(d_adjMatrix, adjMatrix, numNodes * numNodes * sizeof(int), hipMemcpyHostToDevice);

    // Perform BFS
    bfs(d_adjMatrix, numNodes, startNode);

    // Free host and device memory
    delete[] adjMatrix;
    hipFree(d_adjMatrix);

    return 0;
}