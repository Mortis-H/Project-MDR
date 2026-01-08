#include <iostream>
#include <hip/hip_runtime.h>


#define N 16384
#define MAX_EDGES 32768

// Kernel to initialize labels
__global__ void initLabels(int *labels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) labels[idx] = idx;
}

// Kernel to perform one step of the label propagation
__global__ void propagateLabels(int *labels, int *adjList, int *adjListPtr, int *changed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int oldLabel = labels[idx];
        int newLabel = oldLabel;
        for (int i = adjListPtr[idx]; i < adjListPtr[idx + 1]; ++i) {
            newLabel = min(newLabel, labels[adjList[i]]);
        }
        labels[idx] = newLabel;
        if (oldLabel != newLabel) *changed = 1;
    }
}

// Kernel to compress labels
__global__ void compressLabels(int *labels, int *labelMap) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        labels[idx] = labelMap[labels[idx]];
    }
}

// Kernel to build label map
__global__ void buildLabelMap(int *labels, int *labelMap, int *labelCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(&labelCount[labels[idx]], 1);
    }
}

// Kernel to prefix sum for label map
__global__ void prefixSum(int *labelCount, int *labelMap) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        labelMap[idx] = labelCount[idx];
    }
    __syncthreads();
    for (int stride = 1; stride < N; stride *= 2) {
        int val = 0;
        if (idx >= stride) val = labelCount[idx - stride];
        __syncthreads();
        if (idx >= stride) labelCount[idx] += val;
        __syncthreads();
    }
    if (idx < N) {
        labelMap[idx] = labelCount[idx];
        labelCount[idx] = 0;
    }
}

int main() {
    int *adjList, *adjListPtr, *labels, *changed, *labelMap, *labelCount;
    int *d_adjList, *d_adjListPtr, *d_labels, *d_changed, *d_labelMap, *d_labelCount;

    // Allocate memory for host
    adjList = new int[MAX_EDGES];
    adjListPtr = new int[N + 1];
    labels = new int[N];
    changed = new int[1];
    labelMap = new int[N];
    labelCount = new int[N];

    // Initialize adjacency list (example: fully connected for simplicity)
    for (int i = 0; i < N; ++i) {
        adjListPtr[i] = i * 16; // Each node has 16 edges for simplicity
        for (int j = 0; j < 16; ++j) {
            adjList[i * 16 + j] = (i + j) % N;
        }
    }
    adjListPtr[N] = MAX_EDGES;

    // Allocate memory for device
    hipMalloc((void**)&d_adjList, MAX_EDGES * sizeof(int));
    hipMalloc((void**)&d_adjListPtr, (N + 1) * sizeof(int));
    hipMalloc((void**)&d_labels, N * sizeof(int));
    hipMalloc((void**)&d_changed, sizeof(int));
    hipMalloc((void**)&d_labelMap, N * sizeof(int));
    hipMalloc((void**)&d_labelCount, N * sizeof(int));

    // Copy data from host to device
    hipMemcpy(d_adjList, adjList, MAX_EDGES * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_adjListPtr, adjListPtr, (N + 1) * sizeof(int), hipMemcpyHostToDevice);

    // Initialize labels
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    initLabels<<<numBlocks, blockSize>>>(d_labels);
    hipDeviceSynchronize();

    // Main loop for label propagation
    do {
        *changed = 0;
        hipMemcpy(d_changed, changed, sizeof(int), hipMemcpyHostToDevice);
        propagateLabels<<<numBlocks, blockSize>>>(d_labels, d_adjList, d_adjListPtr, d_changed);
        hipMemcpy(changed, d_changed, sizeof(int), hipMemcpyDeviceToHost);
        hipDeviceSynchronize();
    } while (*changed);

    // Build label map
    buildLabelMap<<<numBlocks, blockSize>>>(d_labels, d_labelMap, d_labelCount);
    hipDeviceSynchronize();

    // Prefix sum for label map
    prefixSum<<<1, N>>>(d_labelCount, d_labelMap);
    hipDeviceSynchronize();

    // Compress labels
    compressLabels<<<numBlocks, blockSize>>>(d_labels, d_labelMap);
    hipDeviceSynchronize();

    // Copy results back to host
    hipMemcpy(labels, d_labels, N * sizeof(int), hipMemcpyDeviceToHost);

    // Free device memory
    hipFree(d_adjList);
    hipFree(d_adjListPtr);
    hipFree(d_labels);
    hipFree(d_changed);
    hipFree(d_labelMap);
    hipFree(d_labelCount);

    // Free host memory
    delete[] adjList;
    delete[] adjListPtr;
    delete[] labels;
    delete[] changed;
    delete[] labelMap;
    delete[] labelCount;

    return 0;
}