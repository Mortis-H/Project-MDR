// priority_queue.cu
// A simple CUDA implementation of a priority queue using multiple GPUs.
// This example demonstrates basic multi-GPU scaling for a priority queue with 2048 elements.

#include <iostream>
#include <hip/hip_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

#define NUM_GPUS 2
#define ELEMENTS_PER_GPU (2048 / NUM_GPUS)

__global__ void insertIntoPriorityQueue(int *d_queue, int *d_values, int num_values) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_values) {
        d_queue[idx] = d_values[idx];
    }
}

__global__ void sortPriorityQueue(int *d_queue, int num_elements) {
    // This kernel is a placeholder. Sorting is done using thrust.
}

int main() {
    // Initialize CUDA devices
    hipSetDevice(0);
    int deviceCount;
    hipGetDeviceCount(&deviceCount);
    if (deviceCount < NUM_GPUS) {
        std::cerr << "Error: This example requires at least " << NUM_GPUS << " GPUs." << std::endl;
        return -1;
    }

    // Host data
    thrust::host_vector<int> h_values(2048);
    for (int i = 0; i < 2048; ++i) {
        h_values[i] = rand() % 10000; // Random values for demonstration
    }

    // Device data
    thrust::device_vector<int> d_queue_total(2048);
    thrust::device_vector<int> d_queue[NUM_GPUS];
    thrust::device_vector<int> d_values[NUM_GPUS];

    // Split data across GPUs
    for (int i = 0; i < NUM_GPUS; ++i) {
        hipSetDevice(i);
        d_values[i] = thrust::device_vector<int>(h_values.begin() + i * ELEMENTS_PER_GPU, h_values.begin() + (i + 1) * ELEMENTS_PER_GPU);
        d_queue[i] = thrust::device_vector<int>(ELEMENTS_PER_GPU);
    }

    // Launch kernels to insert values into priority queues on each GPU
    int blockSize = 256;
    int numBlocks = (ELEMENTS_PER_GPU + blockSize - 1) / blockSize;
    for (int i = 0; i < NUM_GPUS; ++i) {
        hipSetDevice(i);
        insertIntoPriorityQueue<<<numBlocks, blockSize>>>(thrust::raw_pointer_cast(d_queue[i].data()), thrust::raw_pointer_cast(d_values[i].data()), ELEMENTS_PER_GPU);
        hipDeviceSynchronize();

        // Check for errors
        hipError_t err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
            return -1;
        }
    }

    // Copy sorted data back to host and combine
    for (int i = 0; i < NUM_GPUS; ++i) {
        hipSetDevice(i);
        thrust::copy(d_queue[i].begin(), d_queue[i].end(), d_queue_total.begin() + i * ELEMENTS_PER_GPU);
    }

    // Sort the combined queue using thrust
    thrust::sort(d_queue_total.begin(), d_queue_total.end());

    // Copy the sorted queue back to the host
    thrust::host_vector<int> h_queue_sorted = d_queue_total;

    // Print sorted queue
    std::cout << "Sorted Priority Queue:" << std::endl;
    for (int i = 0; i < 2048; ++i) {
        std::cout << h_queue_sorted[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}