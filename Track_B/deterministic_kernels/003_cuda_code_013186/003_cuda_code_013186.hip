#include <iostream>
#include <hip/hip_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/unique.h>
#include <thrust/fill.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/equal.h>
#include <thrust/copy.h>
#include <thrust/remove.h>
#include <thrust/execution_policy.h>

#define BLOCK_SIZE 256
#define NUM_NODES 256

__global__ void initializeLabels(int *labels, int *visited) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_NODES) {
        labels[idx] = idx;
        visited[idx] = 0;
    }
}

__global__ void dfsKernel(int *adjList, int *adjListPtrs, int *labels, int *visited, int *stack, int *stackPtr, int *worklist, int *worklistPtr) {
    extern __shared__ int s_labels[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < NUM_NODES) {
        s_labels[threadIdx.x] = labels[idx];
    }
    __syncthreads();

    if (idx < *worklistPtr) {
        int node = worklist[idx];
        if (!visited[node]) {
            visited[node] = 1;
            int start = adjListPtrs[node];
            int end = adjListPtrs[node + 1];
            for (int neighborIdx = start; neighborIdx < end; neighborIdx++) {
                int neighbor = adjList[neighborIdx];
                if (!visited[neighbor]) {
                    stack[atomicAdd(stackPtr, 1)] = neighbor;
                }
            }
            labels[node] = s_labels[threadIdx.x];
        }
    }
}

void dfs(int *adjList, int *adjListPtrs, int *labels, int *visited, int *stack, int *stackPtr, int *worklist, int *worklistPtr) {
    int blockSize = BLOCK_SIZE;
    int numBlocks = (NUM_NODES + blockSize - 1) / blockSize;
    int sharedMemSize = sizeof(int) * blockSize;

    while (*stackPtr > 0) {
        hipMemcpy(worklist, stack, (*stackPtr) * sizeof(int), hipMemcpyDeviceToDevice);
        hipMemcpy(worklistPtr, stackPtr, sizeof(int), hipMemcpyDeviceToDevice);
        thrust::fill(thrust::device, stackPtr, stackPtr + 1, 0);

        dfsKernel<<<numBlocks, blockSize, sharedMemSize>>>(adjList, adjListPtrs, labels, visited, stack, stackPtr, worklist, worklistPtr);
        hipDeviceSynchronize();
    }
}

__global__ void condenseGraph(int *adjList, int *adjListPtrs, int *labels, int *condensedAdjList, int *condensedAdjListPtrs, int *newLabels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_NODES) {
        int label = labels[idx];
        int start = adjListPtrs[idx];
        int end = adjListPtrs[idx + 1];
        for (int neighborIdx = start; neighborIdx < end; neighborIdx++) {
            int neighbor = adjList[neighborIdx];
            int neighborLabel = labels[neighbor];
            if (label != neighborLabel) {
                int pos = atomicAdd(&condensedAdjListPtrs[label + 1], 1);
                condensedAdjList[pos] = neighborLabel;
            }
        }
    }
}

void condense(int *adjList, int *adjListPtrs, int *labels, int *condensedAdjList, int *condensedAdjListPtrs, int *newLabels) {
    int blockSize = BLOCK_SIZE;
    int numBlocks = (NUM_NODES + blockSize - 1) / blockSize;

    thrust::fill(thrust::device, condensedAdjListPtrs, condensedAdjListPtrs + NUM_NODES + 1, 0);
    condenseGraph<<<numBlocks, blockSize>>>(adjList, adjListPtrs, labels, condensedAdjList, condensedAdjListPtrs, newLabels);
    hipDeviceSynchronize();

    thrust::exclusive_scan(thrust::device, condensedAdjListPtrs, condensedAdjListPtrs + NUM_NODES + 1, condensedAdjListPtrs);
}

int main() {
    // Initialize random adjacency list for demonstration
    thrust::host_vector<int> h_adjListPtrs(NUM_NODES + 1);
    thrust::host_vector<int> h_adjList;
    thrust::host_vector<int> h_labels(NUM_NODES);
    thrust::host_vector<int> h_visited(NUM_NODES);
    thrust::host_vector<int> h_stack(NUM_NODES);
    thrust::host_vector<int> h_stackPtr(1, 0);
    thrust::host_vector<int> h_worklist(NUM_NODES);
    thrust::host_vector<int> h_worklistPtr(1, 0);
    thrust::host_vector<int> h_condensedAdjListPtrs(NUM_NODES + 1);
    thrust::host_vector<int> h_condensedAdjList;

    // Example graph: Randomly generate adjacency list
    h_adjListPtrs[0] = 0;
    for (int i = 0; i < NUM_NODES; ++i) {
        int numEdges = rand() % 10; // Random number of edges per node
        h_adjListPtrs[i + 1] = h_adjListPtrs[i] + numEdges;
        for (int j = 0; j < numEdges; ++j) {
            h_adjList.push_back(rand() % NUM_NODES); // Random target node
        }
    }

    thrust::device_vector<int> d_adjList = h_adjList;
    thrust::device_vector<int> d_adjListPtrs = h_adjListPtrs;
    thrust::device_vector<int> d_labels = h_labels;
    thrust::device_vector<int> d_visited = h_visited;
    thrust::device_vector<int> d_stack = h_stack;
    thrust::device_vector<int> d_stackPtr = h_stackPtr;
    thrust::device_vector<int> d_worklist = h_worklist;
    thrust::device_vector<int> d_worklistPtr = h_worklistPtr;
    thrust::device_vector<int> d_condensedAdjListPtrs = h_condensedAdjListPtrs;
    thrust::device_vector<int> d_condensedAdjList(NUM_NODES * NUM_NODES); // Over-allocate

    // Initialize labels and visited arrays
    initializeLabels<<<(NUM_NODES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(thrust::raw_pointer_cast(d_labels.data()), thrust::raw_pointer_cast(d_visited.data()));
    hipDeviceSynchronize();

    // DFS to find strongly connected components
    for (int i = 0; i < NUM_NODES; ++i) {
        int node = i;
        if (!h_visited[node]) {
            d_stack[0] = node;
            d_stackPtr[0] = 1;
            dfs(thrust::raw_pointer_cast(d_adjList.data()), thrust::raw_pointer_cast(d_adjListPtrs.data()), thrust::raw_pointer_cast(d_labels.data()), thrust::raw_pointer_cast(d_visited.data()), thrust::raw_pointer_cast(d_stack.data()), thrust::raw_pointer_cast(d_stackPtr.data()), thrust::raw_pointer_cast(d_worklist.data()), thrust::raw_pointer_cast(d_worklistPtr.data()));
        }
    }

    // Condense graph
    condense(thrust::raw_pointer_cast(d_adjList.data()), thrust::raw_pointer_cast(d_adjListPtrs.data()), thrust::raw_pointer_cast(d_labels.data()), thrust::raw_pointer_cast(d_condensedAdjList.data()), thrust::raw_pointer_cast(d_condensedAdjListPtrs.data()), thrust::raw_pointer_cast(d_labels.data()));

    // Print labels for demonstration
    thrust::copy(d_labels.begin(), d_labels.end(), h_labels.begin());
    std::cout << "Strongly Connected Components Labels:\n";
    for (int i = 0; i < NUM_NODES; ++i) {
        std::cout << i << ": " << h_labels[i] << "\n";
    }

    return 0;
}