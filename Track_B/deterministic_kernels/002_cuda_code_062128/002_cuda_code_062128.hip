#include <iostream>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <vector>

#define N 8192

__device__ __inline__ void warpReduceMin(volatile int* sdata, int tid) {
    sdata[tid] = min(sdata[tid], sdata[tid + 32]);
    sdata[tid] = min(sdata[tid], sdata[tid + 16]);
    sdata[tid] = min(sdata[tid], sdata[tid + 8]);
    sdata[tid] = min(sdata[tid], sdata[tid + 4]);
    sdata[tid] = min(sdata[tid], sdata[tid + 2]);
    sdata[tid] = min(sdata[tid], sdata[tid + 1]);
}

__global__ void buildMinHeap(int* d_arr) {
    extern __shared__ int sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    if (i < N) sdata[tid] = d_arr[i];
    else sdata[tid] = INT_MAX;
    __syncthreads();

    // Build min heap in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid] > sdata[tid + s]) {
                int temp = sdata[tid];
                sdata[tid] = sdata[tid + s];
                sdata[tid + s] = temp;
            }
        }
        __syncthreads();
    }

    // Write back the root of the heap to global memory
    if (tid == 0) d_arr[blockIdx.x] = sdata[0];
}

__global__ void extractMin(int* d_arr, int* d_result) {
    extern __shared__ int sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    if (i < N) sdata[tid] = d_arr[i];
    else sdata[tid] = INT_MAX;
    __syncthreads();

    // Find the minimum value in the shared memory
    if (blockDim.x >= 64) sdata[tid] = min(sdata[tid], sdata[tid + 32]);
    __syncthreads();
    if (blockDim.x >= 32) sdata[tid] = min(sdata[tid], sdata[tid + 16]);
    __syncthreads();
    if (blockDim.x >= 16) sdata[tid] = min(sdata[tid], sdata[tid + 8]);
    __syncthreads();
    if (blockDim.x >= 8) sdata[tid] = min(sdata[tid], sdata[tid + 4]);
    __syncthreads();
    if (blockDim.x >= 4) sdata[tid] = min(sdata[tid], sdata[tid + 2]);
    __syncthreads();
    if (blockDim.x >= 2) sdata[tid] = min(sdata[tid], sdata[tid + 1]);
    __syncthreads();

    // Write the minimum value found by the block to global memory
    if (tid == 0) d_result[blockIdx.x] = sdata[0];
}

int main() {
    int* h_arr = new int[N];
    for (int i = 0; i < N; ++i) h_arr[i] = rand() % 10000;

    int* d_arr, *d_result;
    hipMalloc(&d_arr, N * sizeof(int));
    hipMalloc(&d_result, N * sizeof(int));
    hipMemcpy(d_arr, h_arr, N * sizeof(int), hipMemcpyHostToDevice);

    int blockSize = 128;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // Build min heap
    buildMinHeap<<<numBlocks, blockSize, blockSize * sizeof(int)>>>(d_arr);
    hipDeviceSynchronize();

    // Extract min from heap
    extractMin<<<numBlocks, blockSize, blockSize * sizeof(int)>>>(d_arr, d_result);
    hipDeviceSynchronize();

    int h_result[numBlocks];
    hipMemcpy(h_result, d_result, numBlocks * sizeof(int), hipMemcpyDeviceToHost);

    // Find the minimum element from the results
    int minElement = *std::min_element(h_result, h_result + numBlocks);

    std::cout << "Minimum element in the array: " << minElement << std::endl;

    // Free memory
    hipFree(d_arr);
    hipFree(d_result);
    delete[] h_arr;

    return 0;
}