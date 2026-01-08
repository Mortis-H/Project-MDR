#include <iostream>
#include <hip/hip_runtime.h>


#define MAX_ELEMENTS 16384
#define BLOCK_SIZE 256

__global__ void insertElement(int* d_heap, int* d_size, int value) {
    int idx = atomicAdd(d_size, 1);
    d_heap[idx] = value;
    int parentIdx = (idx - 1) / 2;
    while (idx > 0 && d_heap[parentIdx] < d_heap[idx]) {
        int temp = d_heap[parentIdx];
        d_heap[parentIdx] = d_heap[idx];
        d_heap[idx] = temp;
        idx = parentIdx;
        parentIdx = (idx - 1) / 2;
    }
}

__global__ void extractMax(int* d_heap, int* d_size) {
    int size = atomicSub(d_size, 1);
    if (size > 0) {
        d_heap[0] = d_heap[size];
        int idx = 0;
        while (true) {
            int leftChild = 2 * idx + 1;
            int rightChild = 2 * idx + 2;
            int largest = idx;
            if (leftChild < size && d_heap[leftChild] > d_heap[largest]) {
                largest = leftChild;
            }
            if (rightChild < size && d_heap[rightChild] > d_heap[largest]) {
                largest = rightChild;
            }
            if (largest != idx) {
                int temp = d_heap[idx];
                d_heap[idx] = d_heap[largest];
                d_heap[largest] = temp;
                idx = largest;
            } else {
                break;
            }
        }
    }
}

__global__ void initializeHeap(int* d_heap, int* d_size) {
    *d_size = 0;
}

__global__ void printHeap(int* d_heap, int* d_size) {
    int size = *d_size;
    for (int i = 0; i < size; i++) {
        printf("%d ", d_heap[i]);
    }
    printf("\n");
}

int main() {
    int* d_heap;
    int* d_size;
    int h_size = 0;

    // Allocate memory on the device
    hipError_t err = hipMalloc((void**)&d_heap, MAX_ELEMENTS * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device heap memory: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    err = hipMalloc((void**)&d_size, sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device size memory: " << hipGetErrorString(err) << std::endl;
        hipFree(d_heap);
        return -1;
    }

    // Initialize heap
    initializeHeap<<<1, 1>>>(d_heap, d_size);
    hipDeviceSynchronize();

    // Insert elements into the heap
    for (int i = 0; i < MAX_ELEMENTS; i++) {
        insertElement<<<1, 1>>>(d_heap, d_size, i);
        hipDeviceSynchronize();
    }

    // Extract elements from the heap
    for (int i = 0; i < MAX_ELEMENTS; i++) {
        extractMax<<<1, 1>>>(d_heap, d_size);
        hipDeviceSynchronize();
    }

    // Print the heap (should be empty)
    printHeap<<<1, 1>>>(d_heap, d_size);
    hipDeviceSynchronize();

    // Free device memory
    hipFree(d_heap);
    hipFree(d_size);

    return 0;
}