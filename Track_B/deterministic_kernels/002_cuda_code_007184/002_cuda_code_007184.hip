#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <vector>
#include <random>

#define MAX_QUEUE_SIZE 8192

__device__ int d_queue[MAX_QUEUE_SIZE];
__device__ int d_heap_size;

__device__ void swap(int &a, int &b) {
    int temp = a;
    a = b;
    b = temp;
}

__device__ void heapify_up(int idx) {
    while (idx > 0) {
        int parent = (idx - 1) >> 1;
        if (d_queue[idx] > d_queue[parent]) {
            swap(d_queue[idx], d_queue[parent]);
            idx = parent;
        } else {
            break;
        }
    }
}

__device__ void heapify_down(int idx) {
    while (true) {
        int left_child = (idx << 1) + 1;
        int right_child = left_child + 1;
        int largest = idx;

        if (left_child < d_heap_size && d_queue[left_child] > d_queue[largest]) {
            largest = left_child;
        }
        if (right_child < d_heap_size && d_queue[right_child] > d_queue[largest]) {
            largest = right_child;
        }
        if (largest != idx) {
            swap(d_queue[idx], d_queue[largest]);
            idx = largest;
        } else {
            break;
        }
    }
}

__global__ void insert_into_heap(int *values, int num_values) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_values) {
        int value = values[idx];
        int pos = atomicAdd(&d_heap_size, 1);
        d_queue[pos] = value;
        heapify_up(pos);
    }
}

__global__ void extract_max(int *max_values) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        if (d_heap_size > 0) {
            max_values[0] = d_queue[0];
            d_queue[0] = d_queue[d_heap_size - 1];
            atomicSub(&d_heap_size, 1);
            heapify_down(0);
        }
    }
}

void initialize_queue(int *h_queue, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 100000);
    for (int i = 0; i < size; ++i) {
        h_queue[i] = dis(gen);
    }
}

void checkCudaError(hipError_t err, const char *msg) {
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << msg << " (" << hipGetErrorString(err) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    int *h_queue = new int[MAX_QUEUE_SIZE];
    initialize_queue(h_queue, MAX_QUEUE_SIZE);

    int *d_queue, *d_max_values;
    int *h_max_values = new int[1];

    checkCudaError(hipMalloc(&d_queue, MAX_QUEUE_SIZE * sizeof(int)), "Failed to allocate d_queue");
    checkCudaError(hipMalloc(&d_max_values, sizeof(int)), "Failed to allocate d_max_values");

    int num_threads = 256;
    int num_blocks = (MAX_QUEUE_SIZE + num_threads - 1) / num_threads;

    // Initialize heap size to 0
    checkCudaError(hipMemset(&d_heap_size, 0, sizeof(int)), "Failed to initialize d_heap_size");

    // Insert elements into the heap
    insert_into_heap<<<num_blocks, num_threads>>>(h_queue, MAX_QUEUE_SIZE);
    checkCudaError(hipGetLastError(), "Kernel launch failed for insert_into_heap");
    checkCudaError(hipDeviceSynchronize(), "Failed to synchronize after insert_into_heap");

    // Extract max element
    extract_max<<<1, 1>>>(d_max_values);
    checkCudaError(hipGetLastError(), "Kernel launch failed for extract_max");
    checkCudaError(hipDeviceSynchronize(), "Failed to synchronize after extract_max");

    // Copy result back to host
    checkCudaError(hipMemcpy(h_max_values, d_max_values, sizeof(int), hipMemcpyDeviceToHost), "Failed to copy max value back to host");

    std::cout << "Max value in the heap: " << h_max_values[0] << std::endl;

    // Free device memory
    checkCudaError(hipFree(d_queue), "Failed to free d_queue");
    checkCudaError(hipFree(d_max_values), "Failed to free d_max_values");

    delete[] h_queue;
    delete[] h_max_values;

    return 0;
}