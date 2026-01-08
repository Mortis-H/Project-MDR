// priority_queue.cu
// A simple CUDA implementation of a priority queue using persistent threads
// with 2048 elements, focusing on thread block synchronization.

#include <iostream>
#include <hip/hip_runtime.h>

#define N 2048  // Number of elements in the priority queue

// Kernel to maintain the max heap property
__global__ void maxHeapify(int* d_array, int n) {
    extern __shared__ int s_data[];
    int idx = threadIdx.x;

    // Load data into shared memory
    s_data[idx] = d_array[idx];
    __syncthreads();

    // Build the max heap
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (idx < s) {
            if (s_data[idx] < s_data[idx + s]) {
                int temp = s_data[idx];
                s_data[idx] = s_data[idx + s];
                s_data[idx + s] = temp;
            }
        }
        __syncthreads();
    }

    // If this is the root thread, it holds the max element
    if (idx == 0) {
        d_array[0] = s_data[0];
    }
}

// Kernel to extract the max element and re-heapify
__global__ void extractMax(int* d_array, int n) {
    extern __shared__ int s_data[];
    int idx = threadIdx.x;

    // Load data into shared memory
    s_data[idx] = d_array[idx];
    __syncthreads();

    // Root thread extracts the max element
    if (idx == 0) {
        d_array[0] = s_data[n - 1];  // Move the last element to the root
        s_data[0] = s_data[n - 1];
    }
    __syncthreads();

    // Build the max heap again
    for (int s = 1; s < blockDim.x; s <<= 1) {
        int j = idx - s;
        if (j >= 0 && s_data[j] < s_data[j + s]) {
            int temp = s_data[j];
            s_data[j] = s_data[j + s];
            s_data[j + s] = temp;
        }
        __syncthreads();
    }

    // Write back to global memory
    d_array[idx] = s_data[idx];
}

int main() {
    int h_array[N];
    int* d_array;

    // Initialize array with random values
    for (int i = 0; i < N; ++i) {
        h_array[i] = rand() % 1000;
    }

    // Allocate device memory
    hipError_t err = hipMalloc((void**)&d_array, N * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Error allocating device memory: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy data from host to device
    err = hipMemcpy(d_array, h_array, N * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Error copying data to device: " << hipGetErrorString(err) << std::endl;
        hipFree(d_array);
        return -1;
    }

    // Build the max heap
    maxHeapify<<<1, N, N * sizeof(int)>>>(d_array, N);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error launching maxHeapify kernel: " << hipGetErrorString(err) << std::endl;
        hipFree(d_array);
        return -1;
    }
    hipDeviceSynchronize();

    // Extract the max element
    extractMax<<<1, N, N * sizeof(int)>>>(d_array, N);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error launching extractMax kernel: " << hipGetErrorString(err) << std::endl;
        hipFree(d_array);
        return -1;
    }
    hipDeviceSynchronize();

    // Copy data back to host
    err = hipMemcpy(h_array, d_array, N * sizeof(int), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Error copying data from device: " << hipGetErrorString(err) << std::endl;
        hipFree(d_array);
        return -1;
    }

    // Print the max element
    std::cout << "Max element: " << h_array[0] << std::endl;

    // Free device memory
    hipFree(d_array);

    return 0;
}