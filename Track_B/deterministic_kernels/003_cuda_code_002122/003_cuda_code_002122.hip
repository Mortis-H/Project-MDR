#include <iostream>
#include <hip/hip_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#define NUM_NODES 128
#define MAX_COLOR 10

__device__ int colors[NUM_NODES];

__global__ void initColors() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_NODES) {
        colors[idx] = -1; // Initialize all nodes with no color
    }
}

__global__ void assignColors(int *adjMatrix, int *degrees, int *conflicts, int *maxDegree, int *maxDegreeIdx) {
    extern __shared__ int shared_colors[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= NUM_NODES) return;

    int myColor = 0;
    bool validColor = false;

    while (!validColor) {
        validColor = true;
        for (int j = 0; j < NUM_NODES; j++) {
            if (adjMatrix[idx * NUM_NODES + j] && colors[j] == myColor) {
                validColor = false;
                break;
            }
        }
        if (!validColor) {
            myColor++;
        }
    }

    colors[idx] = myColor;

    // Check for conflicts
    conflicts[idx] = 0;
    for (int j = 0; j < NUM_NODES; j++) {
        if (adjMatrix[idx * NUM_NODES + j] && colors[j] == myColor) {
            conflicts[idx] = 1;
            break;
        }
    }

    // Calculate max degree
    if (degrees[idx] > *maxDegree) {
        atomicMax(maxDegree, degrees[idx]);
        if (*maxDegree == degrees[idx]) {
            atomicExch(maxDegreeIdx, idx);
        }
    }
}

__global__ void resolveConflicts(int *adjMatrix, int *conflicts, int *degrees, int *maxDegreeIdx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_NODES) return;

    if (conflicts[idx]) {
        int myColor = colors[idx] + 1;
        bool validColor = false;

        while (!validColor) {
            validColor = true;
            for (int j = 0; j < NUM_NODES; j++) {
                if (adjMatrix[idx * NUM_NODES + j] && colors[j] == myColor) {
                    validColor = false;
                    break;
                }
            }
            if (!validColor) {
                myColor++;
            }
        }

        colors[idx] = myColor;
    }
}

int main() {
    // Host data
    thrust::host_vector<int> h_adjMatrix(NUM_NODES * NUM_NODES, 0);
    thrust::host_vector<int> h_degrees(NUM_NODES, 0);
    thrust::host_vector<int> h_conflicts(NUM_NODES, 0);
    int h_maxDegree = 0;
    int h_maxDegreeIdx = 0;

    // Simple graph setup (ring graph)
    for (int i = 0; i < NUM_NODES; i++) {
        h_adjMatrix[i * NUM_NODES + (i + 1) % NUM_NODES] = 1;
        h_adjMatrix[i * NUM_NODES + (i - 1 + NUM_NODES) % NUM_NODES] = 1;
        h_degrees[i] = 2;
    }

    // Device data
    thrust::device_vector<int> d_adjMatrix = h_adjMatrix;
    thrust::device_vector<int> d_degrees = h_degrees;
    thrust::device_vector<int> d_conflicts = h_conflicts;
    int *d_maxDegree, *d_maxDegreeIdx;
    hipMalloc(&d_maxDegree, sizeof(int));
    hipMalloc(&d_maxDegreeIdx, sizeof(int));

    // Initialize colors
    int blockSize = 256;
    int numBlocks = (NUM_NODES + blockSize - 1) / blockSize;
    initColors<<<numBlocks, blockSize>>>();
    hipDeviceSynchronize();

    // Assign colors
    assignColors<<<numBlocks, blockSize, blockSize * sizeof(int)>>>(thrust::raw_pointer_cast(d_adjMatrix.data()),
                                                                  thrust::raw_pointer_cast(d_degrees.data()),
                                                                  thrust::raw_pointer_cast(d_conflicts.data()),
                                                                  d_maxDegree, d_maxDegreeIdx);
    hipDeviceSynchronize();

    // Resolve conflicts
    resolveConflicts<<<numBlocks, blockSize>>>(thrust::raw_pointer_cast(d_adjMatrix.data()),
                                              thrust::raw_pointer_cast(d_conflicts.data()),
                                              thrust::raw_pointer_cast(d_degrees.data()),
                                              d_maxDegreeIdx);
    hipDeviceSynchronize();

    // Check for errors
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy results back to host
    thrust::copy(d_conflicts.begin(), d_conflicts.end(), h_conflicts.begin());
    hipMemcpy(&h_maxDegree, d_maxDegree, sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(&h_maxDegreeIdx, d_maxDegreeIdx, sizeof(int), hipMemcpyDeviceToHost);

    // Output results
    std::cout << "Max Degree: " << h_maxDegree << " at Node: " << h_maxDegreeIdx << std::endl;
    std::cout << "Conflicts: " << thrust::reduce(thrust::device, d_conflicts.begin(), d_conflicts.end()) << std::endl;

    // Free device memory
    hipFree(d_maxDegree);
    hipFree(d_maxDegreeIdx);

    return 0;
}