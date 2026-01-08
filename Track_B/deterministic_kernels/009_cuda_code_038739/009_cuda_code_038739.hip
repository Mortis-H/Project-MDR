#include <iostream>
#include <hip/hip_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#define NUM_NODES 4096
#define BLOCK_SIZE 256

__global__ void assignLeaders(int *leaders, int *parents, int *degrees, int numNodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numNodes) {
        if (leaders[idx] == idx) {
            int current = idx;
            while (parents[current] != current) {
                current = parents[current];
            }
            leaders[idx] = current;
        }
    }
}

__global__ void propagateLeaders(int *leaders, int *parents, int numNodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numNodes) {
        leaders[idx] = leaders[parents[idx]];
    }
}

__global__ void compressPaths(int *leaders, int *parents, int numNodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numNodes) {
        int current = idx;
        while (parents[current] != leaders[current]) {
            int next = parents[current];
            parents[current] = leaders[current];
            current = next;
        }
    }
}

__global__ void initializeLeaders(int *leaders, int *parents, int numNodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numNodes) {
        leaders[idx] = idx;
        parents[idx] = idx;
    }
}

__global__ void computeDegrees(int *degrees, int *leaders, int *parents, int numNodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numNodes) {
        if (parents[idx] == idx) {
            degrees[idx] = 0;
        }
    }
}

__global__ void scanDegrees(int *degrees, int numNodes) {
    extern __shared__ int sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    int ai = 0;
    if (i < numNodes) {
        ai = degrees[i];
    }
    sdata[tid] = ai;
    __syncthreads();
    for (int s = 1; s <= blockDim.x / 2; s *= 2) {
        if (tid < blockDim.x / (2 * s)) {
            int ai = sdata[tid * 2 * s];
            int bi = sdata[tid * 2 * s + s];
            sdata[tid * 2 * s] = ai + bi;
        }
        __syncthreads();
    }
    if (tid == 0) {
        degrees[blockIdx.x] = sdata[0];
    }
}

__global__ void assignDegrees(int *degrees, int *leaders, int *parents, int numNodes) {
    extern __shared__ int sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    int ai = 0;
    if (i < numNodes) {
        ai = degrees[i];
    }
    sdata[tid] = ai;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            int ai = sdata[tid];
            int bi = sdata[tid + s];
            sdata[tid] = ai + bi;
        }
        __syncthreads();
    }
    if (tid == 0) {
        degrees[blockIdx.x] = sdata[0];
    }
}

__global__ void computeLeaders(int *leaders, int *parents, int *degrees, int numNodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numNodes) {
        int current = idx;
        while (parents[current] != current) {
            current = parents[current];
        }
        leaders[idx] = current;
    }
}

__global__ void computeSCC(int *leaders, int *parents, int *degrees, int numNodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numNodes) {
        if (parents[idx] == idx) {
            degrees[idx] = 0;
        }
    }
}

int main() {
    // Initialize random graph for demonstration purposes
    thrust::host_vector<int> h_parents(NUM_NODES);
    for (int i = 0; i < NUM_NODES; ++i) {
        h_parents[i] = rand() % NUM_NODES;
    }

    // Allocate device memory
    int *d_parents, *d_leaders, *d_degrees;
    hipMalloc((void **)&d_parents, NUM_NODES * sizeof(int));
    hipMalloc((void **)&d_leaders, NUM_NODES * sizeof(int));
    hipMalloc((void **)&d_degrees, NUM_NODES * sizeof(int));

    // Copy data to device
    hipMemcpy(d_parents, h_parents.data(), NUM_NODES * sizeof(int), hipMemcpyHostToDevice);

    // Initialize leaders and parents
    initializeLeaders<<<(NUM_NODES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_leaders, d_parents, NUM_NODES);
    hipDeviceSynchronize();

    // Iteratively compute SCCs
    for (int iter = 0; iter < 10; ++iter) { // Assuming 10 iterations for demonstration
        assignLeaders<<<(NUM_NODES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_leaders, d_parents, d_degrees, NUM_NODES);
        hipDeviceSynchronize();
        propagateLeaders<<<(NUM_NODES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_leaders, d_parents, NUM_NODES);
        hipDeviceSynchronize();
        compressPaths<<<(NUM_NODES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_leaders, d_parents, NUM_NODES);
        hipDeviceSynchronize();
    }

    // Copy results back to host
    thrust::device_vector<int> d_leaders_host(d_leaders, d_leaders + NUM_NODES);

    // Free device memory
    hipFree(d_parents);
    hipFree(d_leaders);
    hipFree(d_degrees);

    // Output results
    for (int i = 0; i < 10; ++i) { // Print first 10 leaders for demonstration
        std::cout << "Node " << i << " leader: " << d_leaders_host[i] << std::endl;
    }

    return 0;
}