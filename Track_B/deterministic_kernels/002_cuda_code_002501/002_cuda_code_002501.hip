// priority_queue.cu
// CUDA implementation of a priority queue using persistent threads

#include <iostream>
#include <hip/hip_runtime.h>

#define MAX_ELEMENTS 1024

// Global device memory for the priority queue
__device__ int d_queue[MAX_ELEMENTS];
__device__ int d_size = 0;

// Kernel to insert an element into the priority queue
__global__ void insertKernel(int element) {
    int idx = threadIdx.x;
    if (idx == 0) {
        // Insert element into the queue
        int pos = atomicAdd(&d_size, 1);
        if (pos < MAX_ELEMENTS) {
            d_queue[pos] = element;
            // Simple bubble up to maintain heap property
            while (pos > 0 && d_queue[(pos - 1) / 2] < d_queue[pos]) {
                int temp = d_queue[pos];
                d_queue[pos] = d_queue[(pos - 1) / 2];
                d_queue[(pos - 1) / 2] = temp;
                pos = (pos - 1) / 2;
            }
        }
    }
}

// Kernel to extract the maximum element from the priority queue
__global__ void extractMaxKernel() {
    int idx = threadIdx.x;
    if (idx == 0 && d_size > 0) {
        // Extract the max element (root of the heap)
        int maxElement = d_queue[0];
        d_queue[0] = d_queue[d_size - 1];
        atomicSub(&d_size, 1);
        // Simple bubble down to maintain heap property
        int pos = 0;
        while (true) {
            int left = 2 * pos + 1;
            int right = 2 * pos + 2;
            int largest = pos;
            if (left < d_size && d_queue[left] > d_queue[largest]) {
                largest = left;
            }
            if (right < d_size && d_queue[right] > d_queue[largest]) {
                largest = right;
            }
            if (largest != pos) {
                int temp = d_queue[pos];
                d_queue[pos] = d_queue[largest];
                d_queue[largest] = temp;
                pos = largest;
            } else {
                break;
            }
        }
        // Print the extracted max element
        printf("Extracted max element: %d\n", maxElement);
    }
}

int main() {
    // Initialize CUDA
    hipError_t err;

    // Launch insert kernel multiple times to add elements
    for (int i = 0; i < MAX_ELEMENTS; ++i) {
        insertKernel<<<1, 1>>>(i);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "CUDA error after insertKernel launch: " << hipGetErrorString(err) << std::endl;
            return -1;
        }
        hipDeviceSynchronize();
    }

    // Launch extractMax kernel multiple times to remove elements
    for (int i = 0; i < MAX_ELEMENTS; ++i) {
        extractMaxKernel<<<1, 1>>>();
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "CUDA error after extractMaxKernel launch: " << hipGetErrorString(err) << std::endl;
            return -1;
        }
        hipDeviceSynchronize();
    }

    // Cleanup and exit
    return 0;
}