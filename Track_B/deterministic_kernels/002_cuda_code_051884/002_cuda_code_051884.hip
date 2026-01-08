#include <iostream>
#include <hip/hip_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#define N 8192
#define BLOCK_SIZE 256

// Function to check CUDA errors
#define CUDA_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << hipGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Kernel to build a max heap
__global__ void buildMaxHeap(int* d_arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Ensure each thread is a root of a subtree
    for (int k = n / 2 - 1; k >= 0; k--) {
        int parent = k;
        int child;
        while (true) {
            child = 2 * parent + 1;
            if (child >= n) break;
            if (child + 1 < n && d_arr[child] < d_arr[child + 1]) child++;
            if (d_arr[parent] >= d_arr[child]) break;
            // Swap parent and child
            int temp = d_arr[parent];
            d_arr[parent] = d_arr[child];
            d_arr[child] = temp;
            parent = child;
        }
    }
}

// Kernel to extract the maximum element from the heap
__global__ void extractMax(int* d_arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx != 0) return;

    // Swap the root with the last element
    int temp = d_arr[0];
    d_arr[0] = d_arr[n - 1];
    d_arr[n - 1] = temp;

    // Heapify the root
    int parent = 0;
    int child;
    while (true) {
        child = 2 * parent + 1;
        if (child >= n - 1) break;
        if (child + 1 < n - 1 && d_arr[child] < d_arr[child + 1]) child++;
        if (d_arr[parent] >= d_arr[child]) break;
        // Swap parent and child
        temp = d_arr[parent];
        d_arr[parent] = d_arr[child];
        d_arr[child] = temp;
        parent = child;
    }
}

int main() {
    // Initialize host vector with random values
    thrust::host_vector<int> h_arr(N);
    for (int i = 0; i < N; i++) {
        h_arr[i] = rand() % 10000;
    }

    // Allocate device memory
    int* d_arr;
    CUDA_CHECK(hipMalloc(&d_arr, N * sizeof(int)));

    // Copy data from host to device
    CUDA_CHECK(hipMemcpy(d_arr, h_arr.data(), N * sizeof(int), hipMemcpyHostToDevice));

    // Build max heap
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    buildMaxHeap<<<numBlocks, BLOCK_SIZE>>>(d_arr, N);
    CUDA_CHECK(hipDeviceSynchronize());

    // Extract max element (demonstration of priority queue functionality)
    extractMax<<<1, 1>>>(d_arr, N);
    CUDA_CHECK(hipDeviceSynchronize());

    // Copy data back to host
    CUDA_CHECK(hipMemcpy(h_arr.data(), d_arr, N * sizeof(int), hipMemcpyDeviceToHost));

    // Print the maximum element
    std::cout << "Maximum element extracted: " << h_arr[N - 1] << std::endl;

    // Free device memory
    CUDA_CHECK(hipFree(d_arr));

    return 0;
}