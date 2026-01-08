#include <iostream>
#include <hip/hip_runtime.h>

#define N 4096
#define BLOCK_SIZE 256

__device__ int heap[N];
__device__ int heapSize;

__device__ void heapifyUp(int idx) {
    int parent;
    int temp;
    while (idx > 0) {
        parent = (idx - 1) / 2;
        if (heap[parent] < heap[idx]) {
            temp = heap[parent];
            heap[parent] = heap[idx];
            heap[idx] = temp;
            idx = parent;
        } else {
            break;
        }
    }
}

__device__ void heapifyDown(int idx) {
    int left, right, largest, temp;
    while (idx < heapSize) {
        left = 2 * idx + 1;
        right = 2 * idx + 2;
        largest = idx;

        if (left < heapSize && heap[left] > heap[largest]) {
            largest = left;
        }
        if (right < heapSize && heap[right] > heap[largest]) {
            largest = right;
        }
        if (largest != idx) {
            temp = heap[idx];
            heap[idx] = heap[largest];
            heap[largest] = temp;
            idx = largest;
        } else {
            break;
        }
    }
}

__global__ void insertElements(int *d_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        heap[atomicAdd(&heapSize, 1)] = d_elements[idx];
        heapifyUp(idx);
    }
}

__global__ void extractMax() {
    int idx = threadIdx.x;
    if (idx == 0) {
        heap[0] = heap[--heapSize];
        heapifyDown(0);
    }
}

int main() {
    int *h_elements = new int[N];
    int *d_elements;
    hipError_t err;

    // Initialize elements
    for (int i = 0; i < N; ++i) {
        h_elements[i] = rand() % 10000;
    }

    // Allocate device memory
    err = hipMalloc((void **)&d_elements, N * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Error allocating device memory: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy data to device
    err = hipMemcpy(d_elements, h_elements, N * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Error copying data to device: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Initialize heap size
    heapSize = 0;

    // Insert elements into heap
    insertElements<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_elements);
    hipDeviceSynchronize();

    // Extract max elements
    for (int i = 0; i < N; ++i) {
        extractMax<<<1, BLOCK_SIZE>>>();
        hipDeviceSynchronize();
    }

    // Copy result back to host
    err = hipMemcpy(h_elements, d_elements, N * sizeof(int), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Error copying data from device: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Free device memory
    hipFree(d_elements);

    // Print sorted elements (for verification)
    std::cout << "Sorted elements: ";
    for (int i = 0; i < N; ++i) {
        std::cout << h_elements[i] << " ";
    }
    std::cout << std::endl;

    delete[] h_elements;

    return 0;
}