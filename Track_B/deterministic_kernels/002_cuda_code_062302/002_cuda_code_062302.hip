#include <iostream>
#include <hip/hip_runtime.h>
#include <algorithm>

#define NUM_ELEMENTS 32

__device__ __inline__ void swap(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

__device__ __inline__ void heapifyUp(int* pq, int idx) {
    while (idx > 0) {
        int parent = (idx - 1) / 2;
        if (pq[idx] > pq[parent]) {
            swap(&pq[idx], &pq[parent]);
            idx = parent;
        } else {
            break;
        }
    }
}

__device__ __inline__ void heapifyDown(int* pq, int size) {
    int idx = 0;
    while (true) {
        int left = 2 * idx + 1;
        int right = 2 * idx + 2;
        int largest = idx;

        if (left < size && pq[left] > pq[largest]) {
            largest = left;
        }
        if (right < size && pq[right] > pq[largest]) {
            largest = right;
        }
        if (largest != idx) {
            swap(&pq[idx], &pq[largest]);
            idx = largest;
        } else {
            break;
        }
    }
}

__global__ void insertIntoPQ(int* pq, int* size, int value) {
    int tid = threadIdx.x;
    if (tid == 0) {
        pq[*size] = value;
        (*size)++;
        heapifyUp(pq, *size - 1);
    }
}

__global__ void extractMaxFromPQ(int* pq, int* size) {
    int tid = threadIdx.x;
    if (tid == 0) {
        if (*size > 0) {
            pq[0] = pq[*size - 1];
            (*size)--;
            heapifyDown(pq, *size);
        }
    }
}

int main() {
    int* d_pq;
    int* d_size;
    int h_pq[NUM_ELEMENTS] = {0};
    int h_size = 0;

    // Allocate Unified Memory -- accessible from CPU or GPU
    hipError_t err = hipMallocManaged(&d_pq, NUM_ELEMENTS * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate Unified Memory for priority queue (error code " << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }

    err = hipMallocManaged(&d_size, sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate Unified Memory for size (error code " << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Initialize size
    *d_size = h_size;

    // Insert elements into the priority queue
    for (int i = 0; i < NUM_ELEMENTS; i++) {
        insertIntoPQ<<<1, 1>>>(d_pq, d_size, i * 3 + 1);
        hipDeviceSynchronize(); // Ensure the insert is complete before next operation
    }

    // Extract elements from the priority queue
    for (int i = 0; i < NUM_ELEMENTS; i++) {
        extractMaxFromPQ<<<1, 1>>>(d_pq, d_size);
        hipDeviceSynchronize(); // Ensure the extract is complete before next operation
    }

    // Copy data back to host
    hipMemcpy(h_pq, d_pq, NUM_ELEMENTS * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(&h_size, d_size, sizeof(int), hipMemcpyDeviceToHost);

    // Output the results
    std::cout << "Priority Queue after extraction: ";
    for (int i = 0; i < NUM_ELEMENTS; i++) {
        std::cout << h_pq[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Size of priority queue: " << h_size << std::endl;

    // Free Unified Memory
    hipFree(d_pq);
    hipFree(d_size);

    return 0;
}