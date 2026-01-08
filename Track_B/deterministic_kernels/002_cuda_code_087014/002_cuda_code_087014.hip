#include <iostream>
#include <hip/hip_runtime.h>


#define MAX_QUEUE_SIZE 2048

// Kernel to insert elements into the priority queue
__global__ void insertKernel(int* queue, int* heapSize, int value) {
    int idx = threadIdx.x;
    if (idx >= *heapSize) return;

    // Insert value at the end of the heap
    int pos = atomicAdd(heapSize, 1);
    queue[pos] = value;

    // Heapify up
    while (pos > 0) {
        int parent = (pos - 1) / 2;
        if (queue[pos] > queue[parent]) {
            int temp = queue[pos];
            queue[pos] = queue[parent];
            queue[parent] = temp;
            pos = parent;
        } else {
            break;
        }
    }
}

// Kernel to extract the maximum element from the priority queue
__global__ void extractMaxKernel(int* queue, int* heapSize) {
    if (threadIdx.x == 0 && *heapSize > 0) {
        // Replace the root of the heap with the last element
        int last = atomicSub(heapSize, 1) - 1;
        queue[0] = queue[last];

        // Heapify down
        int pos = 0;
        while (true) {
            int left = 2 * pos + 1;
            int right = 2 * pos + 2;
            int largest = pos;

            if (left < *heapSize && queue[left] > queue[largest]) {
                largest = left;
            }
            if (right < *heapSize && queue[right] > queue[largest]) {
                largest = right;
            }
            if (largest != pos) {
                int temp = queue[pos];
                queue[pos] = queue[largest];
                queue[largest] = temp;
                pos = largest;
            } else {
                break;
            }
        }
    }
}

// Main function to demonstrate the usage of the priority queue
int main() {
    int* d_queue;
    int* d_heapSize;
    int h_heapSize = 0;

    // Allocate memory on the device
    hipError_t err = hipMalloc((void**)&d_queue, MAX_QUEUE_SIZE * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate memory for queue: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    err = hipMalloc((void**)&d_heapSize, sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate memory for heap size: " << hipGetErrorString(err) << std::endl;
        hipFree(d_queue);
        return -1;
    }

    // Initialize heap size on the device
    hipMemcpy(d_heapSize, &h_heapSize, sizeof(int), hipMemcpyHostToDevice);

    // Insert elements into the priority queue
    int numElements = 2048;
    for (int i = 0; i < numElements; ++i) {
        int value = rand() % 10000; // Random values between 0 and 9999
        insertKernel<<<1, 2048>>>(d_queue, d_heapSize, value);
        hipDeviceSynchronize(); // Ensure all inserts are complete before next iteration
    }

    // Extract elements from the priority queue
    for (int i = 0; i < numElements; ++i) {
        extractMaxKernel<<<1, 2048>>>(d_queue, d_heapSize);
        hipDeviceSynchronize(); // Ensure all extractions are complete before next iteration
    }

    // Clean up
    hipFree(d_queue);
    hipFree(d_heapSize);

    std::cout << "Priority queue operations completed successfully." << std::endl;
    return 0;
}