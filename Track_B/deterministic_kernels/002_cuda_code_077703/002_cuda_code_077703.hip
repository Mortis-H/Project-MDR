#include <iostream>
#include <hip/hip_runtime.h>

#define N 64

// Define a simple binary heap for priority queue
__device__ int heap[N];
__device__ int heapSize;

// Function to maintain the heap property after insertion
__device__ void heapifyUp(int index) {
    int temp;
    while (index > 0) {
        int parent = (index - 1) / 2;
        if (heap[parent] > heap[index]) {
            // Swap
            temp = heap[parent];
            heap[parent] = heap[index];
            heap[index] = temp;
            index = parent;
        } else {
            break;
        }
    }
}

// Function to maintain the heap property after removal
__device__ void heapifyDown(int index) {
    int temp;
    while (index * 2 + 1 < heapSize) {
        int leftChild = index * 2 + 1;
        int rightChild = index * 2 + 2;
        int smallest = index;

        if (leftChild < heapSize && heap[leftChild] < heap[smallest]) {
            smallest = leftChild;
        }
        if (rightChild < heapSize && heap[rightChild] < heap[smallest]) {
            smallest = rightChild;
        }
        if (smallest != index) {
            // Swap
            temp = heap[smallest];
            heap[smallest] = heap[index];
            heap[index] = temp;
            index = smallest;
        } else {
            break;
        }
    }
}

// Kernel to insert elements into the heap
__global__ void insertElements(int* d_elements, int numElements) {
    int idx = threadIdx.x;
    if (idx < numElements) {
        heap[heapSize + idx] = d_elements[idx];
        heapifyUp(heapSize + idx);
    }
    __syncthreads();
    if (idx == 0) {
        heapSize += numElements;
    }
}

// Kernel to remove the minimum element from the heap
__global__ void removeMin() {
    int idx = threadIdx.x;
    if (idx == 0 && heapSize > 0) {
        heap[0] = heap[heapSize - 1];
        heapSize--;
        heapifyDown(0);
    }
}

int main() {
    int h_elements[N] = {20, 15, 30, 10, 5, 50, 40};
    int numElements = 7;

    int* d_elements;
    hipError_t err = hipMalloc((void**)&d_elements, numElements * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Error allocating memory on device: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    err = hipMemcpy(d_elements, h_elements, numElements * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Error copying memory to device: " << hipGetErrorString(err) << std::endl;
        hipFree(d_elements);
        return -1;
    }

    // Initialize heap size
    heapSize = 0;

    // Insert elements into the heap
    insertElements<<<1, numElements>>>(d_elements, numElements);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error launching insertElements kernel: " << hipGetErrorString(err) << std::endl;
        hipFree(d_elements);
        return -1;
    }

    // Remove the minimum element from the heap
    removeMin<<<1, 1>>>();
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error launching removeMin kernel: " << hipGetErrorString(err) << std::endl;
        hipFree(d_elements);
        return -1;
    }

    // Free device memory
    hipFree(d_elements);

    std::cout << "Priority queue operations completed successfully." << std::endl;

    return 0;
}