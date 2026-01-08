#include <iostream>
#include <hip/hip_runtime.h>

// Define the size of the heap
#define HEAP_SIZE 32

// Kernel to build a max heap
__global__ void buildMaxHeap(int *heap) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= HEAP_SIZE) return;

    // Simple bottom-up approach to build max heap
    int child = idx;
    int parent = (child - 1) / 2;

    while (child > 0 && heap[parent] < heap[child]) {
        // Swap parent and child
        int temp = heap[parent];
        heap[parent] = heap[child];
        heap[child] = temp;

        child = parent;
        parent = (child - 1) / 2;
    }
}

// Kernel to extract the max element from the heap
__global__ void extractMax(int *heap, int *maxElement) {
    if (threadIdx.x == 0) {
        // Extract root element (max element)
        *maxElement = heap[0];
        // Move last element to root
        heap[0] = heap[HEAP_SIZE - 1];
        heap[HEAP_SIZE - 1] = 0; // Mark as extracted

        // Heapify down
        int parent = 0;
        int leftChild = 2 * parent + 1;
        int rightChild = 2 * parent + 2;

        while (leftChild < HEAP_SIZE - 1) {
            int largest = parent;

            if (heap[leftChild] > heap[largest]) {
                largest = leftChild;
            }

            if (rightChild < HEAP_SIZE - 1 && heap[rightChild] > heap[largest]) {
                largest = rightChild;
            }

            if (largest != parent) {
                // Swap parent and largest child
                int temp = heap[parent];
                heap[parent] = heap[largest];
                heap[largest] = temp;

                parent = largest;
                leftChild = 2 * parent + 1;
                rightChild = 2 * parent + 2;
            } else {
                break;
            }
        }
    }
}

int main() {
    int *d_heap, *d_maxElement;
    int h_heap[HEAP_SIZE] = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32};
    int h_maxElement;

    // Allocate memory on the device
    hipError_t err = hipMalloc((void **)&d_heap, HEAP_SIZE * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Error allocating memory for heap: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    err = hipMalloc((void **)&d_maxElement, sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Error allocating memory for max element: " << hipGetErrorString(err) << std::endl;
        hipFree(d_heap);
        return -1;
    }

    // Copy heap data to the device
    err = hipMemcpy(d_heap, h_heap, HEAP_SIZE * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Error copying heap data to device: " << hipGetErrorString(err) << std::endl;
        hipFree(d_heap);
        hipFree(d_maxElement);
        return -1;
    }

    // Launch kernel to build max heap
    int threadsPerBlock = 32;
    int blocksPerGrid = (HEAP_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    buildMaxHeap<<<blocksPerGrid, threadsPerBlock>>>(d_heap);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error launching buildMaxHeap kernel: " << hipGetErrorString(err) << std::endl;
        hipFree(d_heap);
        hipFree(d_maxElement);
        return -1;
    }

    // Launch kernel to extract max element
    extractMax<<<1, 1>>>(d_heap, d_maxElement);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error launching extractMax kernel: " << hipGetErrorString(err) << std::endl;
        hipFree(d_heap);
        hipFree(d_maxElement);
        return -1;
    }

    // Copy max element back to host
    err = hipMemcpy(&h_maxElement, d_maxElement, sizeof(int), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Error copying max element back to host: " << hipGetErrorString(err) << std::endl;
        hipFree(d_heap);
        hipFree(d_maxElement);
        return -1;
    }

    // Output the max element
    std::cout << "Max element extracted from heap: " << h_maxElement << std::endl;

    // Free device memory
    hipFree(d_heap);
    hipFree(d_maxElement);

    return 0;
}