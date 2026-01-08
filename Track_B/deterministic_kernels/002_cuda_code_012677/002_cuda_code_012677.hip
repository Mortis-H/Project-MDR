#include <iostream>
#include <hip/hip_runtime.h>

// Define the size of the priority queue
#define PQ_SIZE 2048

// Kernel to insert elements into the priority queue
__global__ void insertIntoPQ(int *pq, int *elements, int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        // Simple insertion sort logic for demonstration
        int key = elements[idx];
        int j = idx - 1;
        while (j >= 0 && pq[j] < key) {
            pq[j + 1] = pq[j];
            j = j - 1;
        }
        pq[j + 1] = key;
    }
}

// Kernel to extract the maximum element from the priority queue
__global__ void extractMaxFromPQ(int *pq, int *maxElement) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        // Extract the maximum element (first element in the sorted array)
        *maxElement = pq[0];
        // Shift elements to the left to maintain the queue
        for (int i = 0; i < PQ_SIZE - 1; i++) {
            pq[i] = pq[i + 1];
        }
        // Set the last element to a very low value
        pq[PQ_SIZE - 1] = -1;
    }
}

int main() {
    // Initialize host data
    int h_elements[PQ_SIZE];
    for (int i = 0; i < PQ_SIZE; i++) {
        h_elements[i] = rand() % 1000; // Random values between 0 and 999
    }

    // Allocate device memory
    int *d_pq, *d_elements, *d_maxElement;
    hipError_t err = hipMalloc((void **)&d_pq, PQ_SIZE * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for priority queue (error code " << err << ")" << std::endl;
        return -1;
    }
    err = hipMalloc((void **)&d_elements, PQ_SIZE * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for elements (error code " << err << ")" << std::endl;
        return -1;
    }
    err = hipMalloc((void **)&d_maxElement, sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for max element (error code " << err << ")" << std::endl;
        return -1;
    }

    // Copy data from host to device
    err = hipMemcpy(d_elements, h_elements, PQ_SIZE * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy elements to device (error code " << err << ")" << std::endl;
        return -1;
    }

    // Set up execution configuration
    int blockSize = 256;
    int numBlocks = (PQ_SIZE + blockSize - 1) / blockSize;

    // Insert elements into the priority queue
    insertIntoPQ<<<numBlocks, blockSize>>>(d_pq, d_elements, PQ_SIZE);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch insertIntoPQ kernel (error code " << err << ")" << std::endl;
        return -1;
    }

    // Extract the maximum element from the priority queue
    extractMaxFromPQ<<<1, 1>>>(d_pq, d_maxElement);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch extractMaxFromPQ kernel (error code " << err << ")" << std::endl;
        return -1;
    }

    // Copy the maximum element back to the host
    int h_maxElement;
    err = hipMemcpy(&h_maxElement, d_maxElement, sizeof(int), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy max element to host (error code " << err << ")" << std::endl;
        return -1;
    }

    // Print the maximum element
    std::cout << "Maximum element extracted from the priority queue: " << h_maxElement << std::endl;

    // Free device memory
    hipFree(d_pq);
    hipFree(d_elements);
    hipFree(d_maxElement);

    return 0;
}