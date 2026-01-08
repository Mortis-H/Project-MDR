#include <iostream>
#include <hip/hip_runtime.h>


// Define the maximum number of elements in the heap
#define MAX_HEAP_SIZE 512

// Kernel to build a max heap using warp-level parallelism
__global__ void buildMaxHeap(int* d_data) {
    extern __shared__ int s_data[];

    // Load data into shared memory
    int idx = threadIdx.x;
    s_data[idx] = d_data[idx];
    __syncthreads();

    // Build the heap in a bottom-up manner
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (idx < offset) {
            int left = idx * 2 + 1;
            int right = idx * 2 + 2;
            int largest = left;

            // Compare with right child if it exists
            if (right < blockDim.x && s_data[right] > s_data[largest]) {
                largest = right;
            }

            // Compare with parent and swap if necessary
            if (s_data[largest] > s_data[idx]) {
                int temp = s_data[idx];
                s_data[idx] = s_data[largest];
                s_data[largest] = temp;
            }
        }
        __syncthreads();
    }

    // Write the sorted data back to global memory
    d_data[idx] = s_data[idx];
}

// Kernel to extract the maximum element from the heap
__global__ void extractMax(int* d_data, int* d_max) {
    extern __shared__ int s_data[];

    // Load data into shared memory
    int idx = threadIdx.x;
    s_data[idx] = d_data[idx];
    __syncthreads();

    // Extract the maximum element
    if (idx == 0) {
        *d_max = s_data[0];
        s_data[0] = s_data[blockDim.x - 1];
    }
    __syncthreads();

    // Rebuild the heap
    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        if (idx < offset) {
            int left = idx * 2 + 1;
            int right = idx * 2 + 2;
            int largest = idx;

            // Compare with left child
            if (left < blockDim.x && s_data[left] > s_data[largest]) {
                largest = left;
            }

            // Compare with right child
            if (right < blockDim.x && s_data[right] > s_data[largest]) {
                largest = right;
            }

            // Swap with the largest child if necessary
            if (largest != idx) {
                int temp = s_data[idx];
                s_data[idx] = s_data[largest];
                s_data[largest] = temp;
            }
        }
        __syncthreads();
    }

    // Write the data back to global memory
    d_data[idx] = s_data[idx];
}

int main() {
    int h_data[MAX_HEAP_SIZE];
    int h_max;

    // Initialize the data array with random values
    for (int i = 0; i < MAX_HEAP_SIZE; i++) {
        h_data[i] = rand() % 1000;
    }

    int* d_data;
    int* d_max;

    // Allocate memory on the device
    hipError_t err = hipMalloc((void**)&d_data, MAX_HEAP_SIZE * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate memory for d_data: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    err = hipMalloc((void**)&d_max, sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate memory for d_max: " << hipGetErrorString(err) << std::endl;
        hipFree(d_data);
        return -1;
    }

    // Copy data from host to device
    err = hipMemcpy(d_data, h_data, MAX_HEAP_SIZE * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy data to device: " << hipGetErrorString(err) << std::endl;
        hipFree(d_data);
        hipFree(d_max);
        return -1;
    }

    // Build the max heap
    buildMaxHeap<<<1, MAX_HEAP_SIZE, MAX_HEAP_SIZE * sizeof(int)>>>(d_data);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel execution failed: " << hipGetErrorString(err) << std::endl;
        hipFree(d_data);
        hipFree(d_max);
        return -1;
    }

    // Extract the maximum element
    extractMax<<<1, MAX_HEAP_SIZE, MAX_HEAP_SIZE * sizeof(int)>>>(d_data, d_max);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel execution failed: " << hipGetErrorString(err) << std::endl;
        hipFree(d_data);
        hipFree(d_max);
        return -1;
    }

    // Copy the maximum element back to the host
    err = hipMemcpy(&h_max, d_max, sizeof(int), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy data from device: " << hipGetErrorString(err) << std::endl;
        hipFree(d_data);
        hipFree(d_max);
        return -1;
    }

    // Output the maximum element
    std::cout << "Maximum element in the heap: " << h_max << std::endl;

    // Free device memory
    hipFree(d_data);
    hipFree(d_max);

    return 0;
}