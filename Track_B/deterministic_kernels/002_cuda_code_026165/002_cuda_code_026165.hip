// skip_list_cuda.cu
#include <iostream>
#include <hip/hip_runtime.h>

#define NUM_ELEMENTS 32
#define NUM_GPUS 2

__device__ int d_skip_list[NUM_ELEMENTS];
__device__ int d_next[NUM_ELEMENTS];

__global__ void initSkipListKernel(int* d_list, int* d_next_list) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < NUM_ELEMENTS) {
        d_list[idx] = idx; // Simple initialization
        d_next_list[idx] = idx + 1; // Simple next pointer setup
    }
}

__global__ void searchSkipListKernel(int* d_list, int* d_next_list, int search_key, int* d_result) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < NUM_ELEMENTS) {
        int current = idx;
        while (current < NUM_ELEMENTS && d_list[current] < search_key) {
            current = d_next_list[current];
        }
        if (current < NUM_ELEMENTS && d_list[current] == search_key) {
            atomicExch(d_result, current);
        }
    }
}

int main() {
    int h_skip_list[NUM_ELEMENTS];
    int h_next[NUM_ELEMENTS];
    int h_result = -1;

    int* d_skip_list[NUM_GPUS];
    int* d_next[NUM_GPUS];
    int* d_result[NUM_GPUS];

    int search_key = 15;

    hipSetDevice(0);
    hipMalloc(&d_skip_list[0], NUM_ELEMENTS * sizeof(int));
    hipMalloc(&d_next[0], NUM_ELEMENTS * sizeof(int));
    hipMalloc(&d_result[0], sizeof(int));

    hipSetDevice(1);
    hipMalloc(&d_skip_list[1], NUM_ELEMENTS * sizeof(int));
    hipMalloc(&d_next[1], NUM_ELEMENTS * sizeof(int));
    hipMalloc(&d_result[1], sizeof(int));

    for (int i = 0; i < NUM_ELEMENTS; ++i) {
        h_skip_list[i] = i;
        h_next[i] = i + 1;
    }
    h_next[NUM_ELEMENTS - 1] = NUM_ELEMENTS; // Mark end of list

    hipSetDevice(0);
    hipMemcpy(d_skip_list[0], h_skip_list, NUM_ELEMENTS * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_next[0], h_next, NUM_ELEMENTS * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_result[0], &h_result, sizeof(int), hipMemcpyHostToDevice);

    hipSetDevice(1);
    hipMemcpy(d_skip_list[1], h_skip_list, NUM_ELEMENTS * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_next[1], h_next, NUM_ELEMENTS * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_result[1], &h_result, sizeof(int), hipMemcpyHostToDevice);

    dim3 threadsPerBlock(256);
    dim3 numBlocks((NUM_ELEMENTS + threadsPerBlock.x - 1) / threadsPerBlock.x);

    hipSetDevice(0);
    initSkipListKernel<<<numBlocks, threadsPerBlock>>>(d_skip_list[0], d_next[0]);
    hipDeviceSynchronize();

    hipSetDevice(1);
    initSkipListKernel<<<numBlocks, threadsPerBlock>>>(d_skip_list[1], d_next[1]);
    hipDeviceSynchronize();

    hipSetDevice(0);
    searchSkipListKernel<<<numBlocks, threadsPerBlock>>>(d_skip_list[0], d_next[0], search_key, d_result[0]);
    hipDeviceSynchronize();

    hipSetDevice(1);
    searchSkipListKernel<<<numBlocks, threadsPerBlock>>>(d_skip_list[1], d_next[1], search_key, d_result[1]);
    hipDeviceSynchronize();

    hipSetDevice(0);
    hipMemcpy(&h_result, d_result[0], sizeof(int), hipMemcpyDeviceToHost);

    hipSetDevice(1);
    hipMemcpy(&h_result, d_result[1], sizeof(int), hipMemcpyDeviceToHost);

    if (h_result != -1) {
        std::cout << "Key " << search_key << " found at index " << h_result << std::endl;
    } else {
        std::cout << "Key " << search_key << " not found" << std::endl;
    }

    hipSetDevice(0);
    hipFree(d_skip_list[0]);
    hipFree(d_next[0]);
    hipFree(d_result[0]);

    hipSetDevice(1);
    hipFree(d_skip_list[1]);
    hipFree(d_next[1]);
    hipFree(d_result[1]);

    return 0;
}