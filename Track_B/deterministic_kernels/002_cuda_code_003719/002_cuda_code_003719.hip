#include <iostream>
#include <hip/hip_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/fill.h>
#include <thrust/sort.h>
#include <thrust/equal.h>
#include <thrust/copy.h>

// CUDA kernel for computing the transpose of the adjacency matrix
__global__ void transposeKernel(int *adjMatrix, int *adjTranspose, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        adjTranspose[col * n + row] = adjMatrix[row * n + col];
    }
}

// CUDA kernel for finding strongly connected components using the Kosaraju's algorithm
__global__ void dfsKernel(int *adjMatrix, int *visited, int *component, int *stack, int *stackPtr, int n, int node) {
    extern __shared__ int sharedVisited[];
    int tid = threadIdx.x;
    int row = node;

    // Initialize shared memory for visited nodes
    if (tid < n) {
        sharedVisited[tid] = 0;
    }
    __syncthreads();

    // Depth-first search
    while (row != -1) {
        if (!sharedVisited[row]) {
            sharedVisited[row] = 1;
            visited[row] = 1;
            component[row] = node;

            // Push unvisited neighbors onto the stack
            for (int col = 0; col < n; col++) {
                if (adjMatrix[row * n + col] && !visited[col]) {
                    stack[atomicAdd(stackPtr, 1)] = col;
                }
            }
        }
        // Pop the next node from the stack
        row = (stackPtr[0] > 0) ? stack[atomicSub(stackPtr, 1)] : -1;
    }
    __syncthreads();
}

int main() {
    const int n = 256; // Number of nodes
    int *adjMatrix, *adjTranspose, *visited, *component, *stack, *stackPtr;
    int *d_adjMatrix, *d_adjTranspose, *d_visited, *d_component, *d_stack, *d_stackPtr;

    // Allocate memory on host
    adjMatrix = new int[n * n];
    adjTranspose = new int[n * n];
    visited = new int[n];
    component = new int[n];
    stack = new int[n * n];
    stackPtr = new int[1];

    // Initialize adjacency matrix (example graph)
    thrust::host_vector<int> h_adjMatrix(n * n, 0);
    // Example graph: Randomly set some edges
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i != j && (rand() % 10) == 0) { // 10% probability of an edge
                h_adjMatrix[i * n + j] = 1;
            }
        }
    }
    thrust::copy(h_adjMatrix.begin(), h_adjMatrix.end(), adjMatrix);

    // Allocate memory on device
    hipMalloc((void **)&d_adjMatrix, n * n * sizeof(int));
    hipMalloc((void **)&d_adjTranspose, n * n * sizeof(int));
    hipMalloc((void **)&d_visited, n * sizeof(int));
    hipMalloc((void **)&d_component, n * sizeof(int));
    hipMalloc((void **)&d_stack, n * n * sizeof(int));
    hipMalloc((void **)&d_stackPtr, sizeof(int));

    // Copy data from host to device
    hipMemcpy(d_adjMatrix, adjMatrix, n * n * sizeof(int), hipMemcpyHostToDevice);
    thrust::fill_n(visited, n, 0);
    hipMemcpy(d_visited, visited, n * sizeof(int), hipMemcpyHostToDevice);
    thrust::fill_n(component, n, -1);
    hipMemcpy(d_component, component, n * sizeof(int), hipMemcpyHostToDevice);
    thrust::fill_n(stack, n * n, -1);
    hipMemcpy(d_stack, stack, n * n * sizeof(int), hipMemcpyHostToDevice);
    stackPtr[0] = 0;
    hipMemcpy(d_stackPtr, stackPtr, sizeof(int), hipMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (n + blockSize.y - 1) / blockSize.y);

    // Transpose the adjacency matrix
    transposeKernel<<<gridSize, blockSize>>>(d_adjMatrix, d_adjTranspose, n);
    hipDeviceSynchronize();

    // Perform the first DFS pass to fill the stack
    for (int i = 0; i < n; i++) {
        if (!visited[i]) {
            dfsKernel<<<1, n, n * sizeof(int)>>>(d_adjMatrix, d_visited, d_component, d_stack, d_stackPtr, n, i);
            hipDeviceSynchronize();
        }
    }

    // Reset visited array
    thrust::fill_n(visited, n, 0);
    hipMemcpy(d_visited, visited, n * sizeof(int), hipMemcpyHostToDevice);

    // Perform the second DFS pass on the transposed graph
    for (int i = n * n - 1; i >= 0; i--) {
        int node = stack[i];
        if (node != -1 && !visited[node]) {
            dfsKernel<<<1, n, n * sizeof(int)>>>(d_adjTranspose, d_visited, d_component, d_stack, d_stackPtr, n, node);
            hipDeviceSynchronize();
        }
    }

    // Copy results from device to host
    hipMemcpy(component, d_component, n * sizeof(int), hipMemcpyDeviceToHost);

    // Output the results
    std::cout << "Strongly Connected Components:\n";
    for (int i = 0; i < n; i++) {
        std::cout << "Node " << i << " -> Component " << component[i] << "\n";
    }

    // Free device memory
    hipFree(d_adjMatrix);
    hipFree(d_adjTranspose);
    hipFree(d_visited);
    hipFree(d_component);
    hipFree(d_stack);
    hipFree(d_stackPtr);

    // Free host memory
    delete[] adjMatrix;
    delete[] adjTranspose;
    delete[] visited;
    delete[] component;
    delete[] stack;
    delete[] stackPtr;

    return 0;
}