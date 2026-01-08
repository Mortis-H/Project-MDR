#include <iostream>
#include <hip/hip_runtime.h>

// Define the size of the priority queue
#define PQ_SIZE 32

// Kernel to insert elements into the priority queue
__global__ void insertIntoPQ(int* pq, int* elements, int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        // Insert element into the priority queue using a simple bubble sort approach
        for (int i = 0; i < PQ_SIZE; ++i) {
            if (elements[idx] > pq[i]) {
                // Shift elements to the right
                for (int j = PQ_SIZE - 1; j > i; --j) {
                    pq[j] = pq[j - 1];
                }
                pq[i] = elements[idx];
                break;
            }
        }
    }
}

// Kernel to extract the maximum element from the priority queue
__global__ void extractMaxFromPQ(int* pq, int* maxElement) {
    // Only one thread extracts the maximum element
    if (threadIdx.x == 0) {
        *maxElement = pq[0];
        // Shift elements to the left to fill the gap
        for (int i = 0; i < PQ_SIZE - 1; ++i) {
            pq[i] = pq[i + 1];
        }
        pq[PQ_SIZE - 1] = INT_MIN; // Reset the last element
    }
}

int main() {
    // Initialize host data
    int h_elements[] = {23, 17, 14, 6, 13, 12, 9, 5, 8, 7, 4, 3, 2, 1, 15, 10, 11, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32};
    int h_pq[PQ_SIZE];
    int h_maxElement;

    // Initialize priority queue with minimum values
    for (int i = 0; i < PQ_SIZE; ++i) {
        h_pq[i] = INT_MIN;
    }

    // Allocate device memory
    int* d_pq;
    int* d_elements;
    int* d_maxElement;
    hipError_t err = hipMalloc((void**)&d_pq, PQ_SIZE * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for priority queue" << std::endl;
        return -1;
    }
    err = hipMalloc((void**)&d_elements, sizeof(h_elements));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for elements" << std::endl;
        return -1;
    }
    err = hipMalloc((void**)&d_maxElement, sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for max element" << std::endl;
        return -1;
    }

    // Copy data to device
    err = hipMemcpy(d_elements, h_elements, sizeof(h_elements), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy elements to device" << std::endl;
        return -1;
    }
    err = hipMemcpy(d_pq, h_pq, PQ_SIZE * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy priority queue to device" << std::endl;
        return -1;
    }

    // Define block and grid sizes
    dim3 blockSize(256);
    dim3 gridSize((sizeof(h_elements) / sizeof(int) + blockSize.x - 1) / blockSize.x);

    // Insert elements into the priority queue
    insertIntoPQ<<<gridSize, blockSize>>>(d_pq, d_elements, sizeof(h_elements) / sizeof(int));
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel launch failed: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Extract the maximum element from the priority queue
    extractMaxFromPQ<<<1, blockSize>>>(d_pq, d_maxElement);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel launch failed: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy results back to host
    err = hipMemcpy(h_pq, d_pq, PQ_SIZE * sizeof(int), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy priority queue back to host" << std::endl;
        return -1;
    }
    err = hipMemcpy(&h_maxElement, d_maxElement, sizeof(int), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy max element back to host" << std::endl;
        return -1;
    }

    // Free device memory
    hipFree(d_pq);
    hipFree(d_elements);
    hipFree(d_maxElement);

    // Print the maximum element extracted
    std::cout << "Maximum element extracted from the priority queue: " << h_maxElement << std::endl;

    return 0;
}