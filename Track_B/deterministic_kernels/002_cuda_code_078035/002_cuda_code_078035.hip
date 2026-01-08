#include <iostream>
#include <hip/hip_runtime.h>

#include <algorithm>

// Define the size of the priority queue
#define PQ_SIZE 256

// Kernel to insert elements into the priority queue
__global__ void insertIntoPQ(int* pq, int* elements, int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        // Insert element into the priority queue using binary heap property
        int currentIdx = idx;
        while (currentIdx > 0) {
            int parentIdx = (currentIdx - 1) / 2;
            if (pq[parentIdx] < elements[idx]) {
                // Swap elements
                int temp = pq[parentIdx];
                pq[parentIdx] = elements[idx];
                pq[currentIdx] = temp;
                currentIdx = parentIdx;
            } else {
                break;
            }
        }
    }
}

// Kernel to extract the maximum element from the priority queue
__global__ void extractMaxFromPQ(int* pq, int* maxElement) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        // Only one thread extracts the max element
        *maxElement = pq[0];
        // Move the last element to the root and heapify
        pq[0] = pq[PQ_SIZE - 1];
        int currentIdx = 0;
        while (true) {
            int leftChildIdx = 2 * currentIdx + 1;
            int rightChildIdx = 2 * currentIdx + 2;
            int largestIdx = currentIdx;

            if (leftChildIdx < PQ_SIZE - 1 && pq[leftChildIdx] > pq[largestIdx]) {
                largestIdx = leftChildIdx;
            }
            if (rightChildIdx < PQ_SIZE - 1 && pq[rightChildIdx] > pq[largestIdx]) {
                largestIdx = rightChildIdx;
            }
            if (largestIdx != currentIdx) {
                // Swap elements
                int temp = pq[largestIdx];
                pq[largestIdx] = pq[currentIdx];
                pq[currentIdx] = temp;
                currentIdx = largestIdx;
            } else {
                break;
            }
        }
    }
}

int main() {
    // Number of elements to insert
    int numElements = PQ_SIZE;
    int* h_elements = new int[numElements];
    int* h_pq = new int[PQ_SIZE];
    int* h_maxElement = new int[1];

    // Initialize elements with random values
    for (int i = 0; i < numElements; i++) {
        h_elements[i] = rand() % 1000;
    }

    // Allocate unified memory for priority queue and elements
    int* d_pq;
    int* d_elements;
    int* d_maxElement;
    hipMallocManaged(&d_pq, PQ_SIZE * sizeof(int));
    hipMallocManaged(&d_elements, numElements * sizeof(int));
    hipMallocManaged(&d_maxElement, sizeof(int));

    // Copy elements to device
    hipMemcpy(d_elements, h_elements, numElements * sizeof(int), hipMemcpyHostToDevice);

    // Set up execution configuration
    int blockSize = 256;
    int numBlocks = (numElements + blockSize - 1) / blockSize;

    // Insert elements into the priority queue
    insertIntoPQ<<<numBlocks, blockSize>>>(d_pq, d_elements, numElements);
    hipDeviceSynchronize();

    // Extract the maximum element from the priority queue
    extractMaxFromPQ<<<1, 1>>>(d_pq, d_maxElement);
    hipDeviceSynchronize();

    // Copy results back to host
    hipMemcpy(h_pq, d_pq, PQ_SIZE * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(h_maxElement, d_maxElement, sizeof(int), hipMemcpyDeviceToHost);

    // Output the maximum element
    std::cout << "Maximum element extracted from priority queue: " << *h_maxElement << std::endl;

    // Free memory
    hipFree(d_pq);
    hipFree(d_elements);
    hipFree(d_maxElement);
    delete[] h_elements;
    delete[] h_pq;
    delete[] h_maxElement;

    return 0;
}