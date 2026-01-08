#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <vector>

namespace cg = cooperative_groups;

// Define the maximum number of elements in the heap
const int MAX_HEAP_SIZE = 4096;

// Kernel to build a max heap
__global__ void buildMaxHeap(int* d_heap, int n) {
    extern __shared__ int s_heap[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    if (i < n) {
        s_heap[tid] = d_heap[i];
    } else {
        s_heap[tid] = INT_MIN;
    }
    __syncthreads();

    // Build heap in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_heap[tid] < s_heap[tid + s]) {
                int temp = s_heap[tid];
                s_heap[tid] = s_heap[tid + s];
                s_heap[tid + s] = temp;
            }
        }
        __syncthreads();
    }

    // Write back to global memory
    if (i < n) {
        d_heap[i] = s_heap[tid];
    }
}

// Kernel to heapify a subtree rooted with node i
__global__ void heapify(int* d_heap, int n, int i) {
    extern __shared__ int s_heap[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    if (idx < n) {
        s_heap[tid] = d_heap[idx];
    } else {
        s_heap[tid] = INT_MIN;
    }
    __syncthreads();

    // Heapify process
    while (i < n) {
        int largest = i;
        int left = 2 * i + 1;
        int right = 2 * i + 2;

        if (left < n && s_heap[left] > s_heap[largest]) {
            largest = left;
        }
        if (right < n && s_heap[right] > s_heap[largest]) {
            largest = right;
        }

        if (largest != i) {
            int temp = s_heap[i];
            s_heap[i] = s_heap[largest];
            s_heap[largest] = temp;

            i = largest;
        } else {
            break;
        }
        __syncthreads();
    }

    // Write back to global memory
    if (idx < n) {
        d_heap[idx] = s_heap[tid];
    }
}

// Main function to demonstrate heap operations
int main() {
    int* d_heap;
    int h_heap[MAX_HEAP_SIZE];

    // Initialize heap with random values
    for (int i = 0; i < MAX_HEAP_SIZE; ++i) {
        h_heap[i] = rand() % 1000;
    }

    // Allocate device memory
    hipError_t err = hipMalloc((void**)&d_heap, MAX_HEAP_SIZE * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory" << std::endl;
        return -1;
    }

    // Copy data from host to device
    err = hipMemcpy(d_heap, h_heap, MAX_HEAP_SIZE * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy data to device" << std::endl;
        hipFree(d_heap);
        return -1;
    }

    // Define block and grid sizes
    int blockSize = 256;
    int gridSize = (MAX_HEAP_SIZE + blockSize - 1) / blockSize;

    // Build max heap
    buildMaxHeap<<<gridSize, blockSize, blockSize * sizeof(int)>>>(d_heap, MAX_HEAP_SIZE);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel launch failed: " << hipGetErrorString(err) << std::endl;
        hipFree(d_heap);
        return -1;
    }
    hipDeviceSynchronize();

    // Heapify to ensure max heap property
    heapify<<<gridSize, blockSize, blockSize * sizeof(int)>>>(d_heap, MAX_HEAP_SIZE, 0);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel launch failed: " << hipGetErrorString(err) << std::endl;
        hipFree(d_heap);
        return -1;
    }
    hipDeviceSynchronize();

    // Copy data back to host
    err = hipMemcpy(h_heap, d_heap, MAX_HEAP_SIZE * sizeof(int), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy data from device" << std::endl;
        hipFree(d_heap);
        return -1;
    }

    // Print the heap
    std::cout << "Heap after building and heapifying:" << std::endl;
    for (int i = 0; i < 10; ++i) { // Print first 10 elements for brevity
        std::cout << h_heap[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    hipFree(d_heap);

    return 0;
}