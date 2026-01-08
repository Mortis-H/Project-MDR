// priority_queue.cu
// A simple CUDA implementation of a priority queue using dynamic parallelism.
// This example is designed for the Hopper architecture and demonstrates multi-GPU usage.

#include <iostream>
#include <hip/hip_runtime.h>


#define QUEUE_SIZE 64
#define NUM_GPUS 2

// Kernel to initialize the priority queue
__global__ void initQueue(int *queue) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < QUEUE_SIZE) {
        queue[idx] = -1; // Initialize with -1 to indicate empty slots
    }
}

// Kernel to insert an element into the priority queue
__global__ void insertElement(int *queue, int element) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < QUEUE_SIZE) {
        if (queue[idx] == -1) {
            queue[idx] = element;
            return;
        }
    }
}

// Kernel to extract the maximum element from the priority queue
__global__ void extractMax(int *queue, int *maxElement) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ int localMax[QUEUE_SIZE];
    localMax[threadIdx.x] = (idx < QUEUE_SIZE) ? queue[idx] : -1;
    __syncthreads();

    // Reduce to find the maximum element
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            localMax[threadIdx.x] = max(localMax[threadIdx.x], localMax[threadIdx.x + s]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicMax(maxElement, localMax[0]);
    }
}

// Kernel to remove the maximum element from the priority queue
__global__ void removeMax(int *queue, int maxElement) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < QUEUE_SIZE) {
        if (queue[idx] == maxElement) {
            queue[idx] = -1;
        }
    }
}

int main() {
    int *d_queue[NUM_GPUS];
    int *d_maxElement[NUM_GPUS];
    int h_maxElement[NUM_GPUS];
    int h_queue[NUM_GPUS][QUEUE_SIZE];
    int numDevices;
    hipGetDeviceCount(&numDevices);
    if (numDevices < NUM_GPUS) {
        std::cerr << "Not enough CUDA devices available." << std::endl;
        return -1;
    }

    for (int gpu = 0; gpu < NUM_GPUS; ++gpu) {
        hipSetDevice(gpu);
        hipError_t err = hipMalloc(&d_queue[gpu], QUEUE_SIZE * sizeof(int));
        if (err != hipSuccess) {
            std::cerr << "Failed to allocate memory on GPU " << gpu << ": " << hipGetErrorString(err) << std::endl;
            return -1;
        }
        err = hipMalloc(&d_maxElement[gpu], sizeof(int));
        if (err != hipSuccess) {
            std::cerr << "Failed to allocate memory on GPU " << gpu << ": " << hipGetErrorString(err) << std::endl;
            return -1;
        }

        // Initialize queue
        initQueue<<<(QUEUE_SIZE + 255) / 256, 256>>>(d_queue[gpu]);
        hipDeviceSynchronize();
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Failed to initialize queue on GPU " << gpu << ": " << hipGetErrorString(err) << std::endl;
            return -1;
        }

        // Insert elements into the queue
        for (int i = 0; i < QUEUE_SIZE; ++i) {
            insertElement<<<1, 1>>>(d_queue[gpu], i);
            hipDeviceSynchronize();
            err = hipGetLastError();
            if (err != hipSuccess) {
                std::cerr << "Failed to insert element on GPU " << gpu << ": " << hipGetErrorString(err) << std::endl;
                return -1;
            }
        }

        // Extract the maximum element
        hipMemset(d_maxElement[gpu], -1, sizeof(int));
        extractMax<<<(QUEUE_SIZE + 255) / 256, 256>>>(d_queue[gpu], d_maxElement[gpu]);
        hipDeviceSynchronize();
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Failed to extract max element on GPU " << gpu << ": " << hipGetErrorString(err) << std::endl;
            return -1;
        }
        hipMemcpy(&h_maxElement[gpu], d_maxElement[gpu], sizeof(int), hipMemcpyDeviceToHost);
        std::cout << "Max element on GPU " << gpu << ": " << h_maxElement[gpu] << std::endl;

        // Remove the maximum element
        removeMax<<<(QUEUE_SIZE + 255) / 256, 256>>>(d_queue[gpu], h_maxElement[gpu]);
        hipDeviceSynchronize();
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Failed to remove max element on GPU " << gpu << ": " << hipGetErrorString(err) << std::endl;
            return -1;
        }
    }

    // Free allocated memory
    for (int gpu = 0; gpu < NUM_GPUS; ++gpu) {
        hipSetDevice(gpu);
        hipFree(d_queue[gpu]);
        hipFree(d_maxElement[gpu]);
    }

    return 0;
}