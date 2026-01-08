#include <iostream>
#include <hip/hip_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#define BLOCK_SIZE 256

// Kernel to build a max heap in shared memory
__global__ void buildMaxHeapKernel(int *d_data, int n) {
    extern __shared__ int s_data[];

    int tid = threadIdx.x;
    s_data[tid] = d_data[tid];
    __syncthreads();

    // Build heap
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_data[tid] < s_data[tid + s]) {
                int temp = s_data[tid];
                s_data[tid] = s_data[tid + s];
                s_data[tid + s] = temp;
            }
        }
        __syncthreads();
    }

    // Write the root of the heap (max element) back to global memory
    if (tid == 0) {
        d_data[blockIdx.x] = s_data[0];
    }
}

// Kernel to extract the max element and rebuild the heap
__global__ void extractMaxKernel(int *d_data, int n, int *d_max) {
    extern __shared__ int s_data[];

    int tid = threadIdx.x;
    s_data[tid] = d_data[tid];
    __syncthreads();

    // Move the last element to the root and rebuild the heap
    if (tid == 0) {
        s_data[0] = s_data[blockDim.x - 1];
        *d_max = s_data[0];
    }
    __syncthreads();

    for (int s = 1; s < blockDim.x; s *= 2) {
        int parent = tid;
        int child = tid * 2 + 1;

        if (child < blockDim.x) {
            if (child + 1 < blockDim.x && s_data[child] < s_data[child + 1]) {
                child++;
            }
            if (s_data[parent] < s_data[child]) {
                int temp = s_data[parent];
                s_data[parent] = s_data[child];
                s_data[child] = temp;
            }
        }
        __syncthreads();
    }

    // Write the new root back to global memory
    if (tid == 0) {
        d_data[blockIdx.x] = s_data[0];
    }
}

int main() {
    int n = BLOCK_SIZE;
    thrust::host_vector<int> h_data(n);

    // Initialize data with random values
    for (int i = 0; i < n; i++) {
        h_data[i] = rand() % 1000;
    }

    thrust::device_vector<int> d_data(h_data);
    int *d_max;
    hipMalloc((void**)&d_max, sizeof(int));

    // Build the max heap
    buildMaxHeapKernel<<<1, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(thrust::raw_pointer_cast(d_data.data()), n);
    hipDeviceSynchronize();

    // Extract the max element and rebuild the heap
    extractMaxKernel<<<1, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(thrust::raw_pointer_cast(d_data.data()), n, d_max);
    hipDeviceSynchronize();

    // Copy the max element back to host
    int max_element;
    hipMemcpy(&max_element, d_max, sizeof(int), hipMemcpyDeviceToHost);

    // Output the max element
    std::cout << "Max element in the priority queue: " << max_element << std::endl;

    // Free device memory
    hipFree(d_max);

    return 0;
}