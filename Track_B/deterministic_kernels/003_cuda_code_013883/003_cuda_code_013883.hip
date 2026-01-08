#include <iostream>
#include <hip/hip_runtime.h>


#define NUM_ELEMENTS 512
#define MAX_GPUS 4

__device__ int d_heap[NUM_ELEMENTS];
__device__ int d_heap_size;

// Device function to maintain heap property
__device__ void heapify_down(int idx) {
    int left, right, smallest;
    while (true) {
        left = 2 * idx + 1;
        right = 2 * idx + 2;
        smallest = idx;

        if (left < d_heap_size && d_heap[left] < d_heap[smallest])
            smallest = left;
        if (right < d_heap_size && d_heap[right] < d_heap[smallest])
            smallest = right;
        if (smallest == idx)
            break;

        int temp = d_heap[idx];
        d_heap[idx] = d_heap[smallest];
        d_heap[smallest] = temp;
        idx = smallest;
    }
}

// Kernel to build heap
__global__ void build_heap(int* data, int n) {
    extern __shared__ int s_data[];
    int tid = threadIdx.x;

    // Load data into shared memory
    if (tid < n)
        s_data[tid] = data[tid];
    __syncthreads();

    // Build heap in shared memory
    for (int i = tid; i < n; i += blockDim.x) {
        int idx = i;
        while (idx > 0 && s_data[(idx - 1) / 2] > s_data[idx]) {
            int parent = (idx - 1) / 2;
            int temp = s_data[parent];
            s_data[parent] = s_data[idx];
            s_data[idx] = temp;
            idx = parent;
        }
    }
    __syncthreads();

    // Copy heap from shared memory to global memory
    if (tid < n)
        d_heap[tid] = s_data[tid];
    if (tid == 0)
        d_heap_size = n;
}

// Kernel to insert element into heap
__global__ void insert_element(int element) {
    int idx = atomicAdd(&d_heap_size, 1);
    d_heap[idx] = element;

    // Heapify up
    while (idx > 0 && d_heap[(idx - 1) / 2] > d_heap[idx]) {
        int parent = (idx - 1) / 2;
        int temp = d_heap[parent];
        d_heap[parent] = d_heap[idx];
        d_heap[idx] = temp;
        idx = parent;
    }
}

// Kernel to extract minimum element from heap
__global__ void extract_min() {
    if (d_heap_size > 0) {
        d_heap[0] = d_heap[d_heap_size - 1];
        atomicSub(&d_heap_size, 1);
        heapify_down(0);
    }
}

// Main function to demonstrate usage
int main() {
    int h_data[NUM_ELEMENTS];
    int* d_data;
    int num_gpus;
    hipGetDeviceCount(&num_gpus);
    num_gpus = std::min(num_gpus, MAX_GPUS);

    // Initialize data
    for (int i = 0; i < NUM_ELEMENTS; i++)
        h_data[i] = rand() % 1000;

    // Allocate device memory
    hipError_t err = hipMalloc(&d_data, NUM_ELEMENTS * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy data to device
    err = hipMemcpy(d_data, h_data, NUM_ELEMENTS * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy data to device: " << hipGetErrorString(err) << std::endl;
        hipFree(d_data);
        return -1;
    }

    // Build heap
    int blockSize = 256;
    int gridSize = (NUM_ELEMENTS + blockSize - 1) / blockSize;
    build_heap<<<gridSize, blockSize, NUM_ELEMENTS * sizeof(int)>>>(d_data, NUM_ELEMENTS);
    hipDeviceSynchronize();

    // Insert new element
    int new_element = 50;
    insert_element<<<1, 1>>>(new_element);
    hipDeviceSynchronize();

    // Extract minimum element
    extract_min<<<1, 1>>>();
    hipDeviceSynchronize();

    // Copy result back to host
    err = hipMemcpy(h_data, d_data, NUM_ELEMENTS * sizeof(int), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy data from device: " << hipGetErrorString(err) << std::endl;
        hipFree(d_data);
        return -1;
    }

    // Print result
    std::cout << "Heap after operations:" << std::endl;
    for (int i = 0; i < NUM_ELEMENTS; i++)
        std::cout << h_data[i] << " ";
    std::cout << std::endl;

    // Free device memory
    hipFree(d_data);
    return 0;
}