#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

// Define the size of the array
#define N 4096

// Kernel to perform partial reduction
__global__ void partial_reduction(float *data, float *shared_data, int size) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    sdata[tid] = (i < size) ? data[i] : 0.0f;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write the result for this block to global memory
    if (tid == 0) {
        shared_data[blockIdx.x] = sdata[0];
    }
}

// Kernel to find the percentile value
__global__ void find_percentile(float *data, int size, float percentile, float *result) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    sdata[tid] = (i < size) ? data[i] : 0.0f;
    __syncthreads();

    // Sort the data in shared memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = 2 * stride * tid;
        float a = (index < blockDim.x) ? sdata[index] : 0.0f;
        float b = (index + stride < blockDim.x) ? sdata[index + stride] : 0.0f;
        if (index < blockDim.x) {
            sdata[index] = (a < b) ? a : b;
        }
        if (index + stride < blockDim.x) {
            sdata[index + stride] = (a > b) ? a : b;
        }
        __syncthreads();
    }

    // Perform bitonic merge sort
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        for (int j = stride; j > 0; j >>= 1) {
            int ixj = tid ^ j;
            if (ixj > tid) {
                if ((tid & stride) == 0) {
                    sdata[tid] = (sdata[tid] < sdata[ixj]) ? sdata[tid] : sdata[ixj];
                } else {
                    sdata[tid] = (sdata[tid] > sdata[ixj]) ? sdata[tid] : sdata[ixj];
                }
            }
            __syncthreads();
        }
    }

    // Write the result for this block to global memory
    if (tid == 0) {
        result[blockIdx.x] = sdata[0];
    }
}

// Function to perform parallel reduction
void parallel_reduction(float *d_data, float *d_shared_data, int size) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    partial_reduction<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_data, d_shared_data, size);
    hipDeviceSynchronize();
}

// Function to find the percentile using CUDA Graphs
void find_percentile_with_graphs(float *d_data, int size, float percentile, float *d_result) {
    hipGraph_t graph;
    hipGraphExec_t graphExec;
    hipGraphNode_t node1, node2;

    hipStream_t streamForGraph;
    hipStreamCreate(&streamForGraph);

    hipStreamBeginCapture(streamForGraph, hipStreamCaptureModeGlobal);

    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    partial_reduction<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_data, d_data, size);
    find_percentile<<<1, blockSize, blockSize * sizeof(float)>>>(d_data, numBlocks, percentile, d_result);

    hipStreamEndCapture(streamForGraph, &graph);

    hipGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

    // Launch the graph
    hipGraphLaunch(graphExec, streamForGraph);
    hipStreamSynchronize(streamForGraph);

    hipGraphDestroy(graph);
    hipGraphExecDestroy(graphExec);
    hipStreamDestroy(streamForGraph);
}

int main() {
    // Initialize data
    thrust::host_vector<float> h_data(N);
    for (int i = 0; i < N; i++) {
        h_data[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Copy data to device
    thrust::device_vector<float> d_data(h_data);
    thrust::device_vector<float> d_shared_data((N + 255) / 256);
    thrust::device_vector<float> d_result(1);

    // Perform parallel reduction
    parallel_reduction(thrust::raw_pointer_cast(d_data.data()), thrust::raw_pointer_cast(d_shared_data.data()), N);

    // Find the 50th percentile
    float percentile = 0.5f;
    find_percentile_with_graphs(thrust::raw_pointer_cast(d_data.data()), N, percentile, thrust::raw_pointer_cast(d_result.data()));

    // Copy result back to host
    float result;
    hipMemcpy(&result, thrust::raw_pointer_cast(d_result.data()), sizeof(float), hipMemcpyDeviceToHost);

    // Print the result
    std::cout << "The " << (percentile * 100) << "th percentile is: " << result << std::endl;

    return 0;
}