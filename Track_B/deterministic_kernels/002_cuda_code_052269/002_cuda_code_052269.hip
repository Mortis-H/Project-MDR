#include <iostream>
#include <hip/hip_runtime.h>


#define MAX_QUEUE_SIZE 32

__global__ void insertIntoPriorityQueue(int* d_queue, int* d_values, int numValues) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numValues) {
        int value = d_values[idx];
        int pos = MAX_QUEUE_SIZE - 1;

        // Shift elements to the right to make space for the new element
        while (pos > 0 && d_queue[pos - 1] < value) {
            d_queue[pos] = d_queue[pos - 1];
            pos--;
        }
        d_queue[pos] = value;
    }
}

__global__ void extractFromPriorityQueue(int* d_queue, int* d_result, int numExtracts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numExtracts) {
        // Extract the maximum element (at index 0) and shift the rest of the queue
        d_result[idx] = d_queue[0];
        for (int i = 0; i < MAX_QUEUE_SIZE - 1; i++) {
            d_queue[i] = d_queue[i + 1];
        }
        d_queue[MAX_QUEUE_SIZE - 1] = 0; // Fill the last position with a default value
    }
}

void checkCudaError(hipError_t err, const char* msg) {
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << msg << " (" << hipGetErrorString(err) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    int numGPUs;
    hipGetDeviceCount(&numGPUs);
    if (numGPUs < 2) {
        std::cerr << "This example requires at least 2 GPUs." << std::endl;
        return EXIT_FAILURE;
    }

    int h_queue[MAX_QUEUE_SIZE] = {0};
    int h_values[MAX_QUEUE_SIZE] = {10, 20, 5, 60, 15, 30, 45, 55, 1, 50, 40, 25, 35, 70, 80, 90, 100, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    int h_result[MAX_QUEUE_SIZE] = {0};

    int* d_queue[numGPUs];
    int* d_values[numGPUs];
    int* d_result[numGPUs];

    int blockSize = 256;
    int numBlocks = (MAX_QUEUE_SIZE + blockSize - 1) / blockSize;

    for (int gpu = 0; gpu < numGPUs; ++gpu) {
        hipSetDevice(gpu);

        // Allocate memory on the device
        checkCudaError(hipMalloc(&d_queue[gpu], MAX_QUEUE_SIZE * sizeof(int)), "hipMalloc d_queue");
        checkCudaError(hipMalloc(&d_values[gpu], MAX_QUEUE_SIZE * sizeof(int)), "hipMalloc d_values");
        checkCudaError(hipMalloc(&d_result[gpu], MAX_QUEUE_SIZE * sizeof(int)), "hipMalloc d_result");

        // Copy data from host to device
        checkCudaError(hipMemcpy(d_values[gpu], h_values, MAX_QUEUE_SIZE * sizeof(int), hipMemcpyHostToDevice), "hipMemcpy d_values");
        checkCudaError(hipMemcpy(d_queue[gpu], h_queue, MAX_QUEUE_SIZE * sizeof(int), hipMemcpyHostToDevice), "hipMemcpy d_queue");

        // Insert values into the priority queue
        insertIntoPriorityQueue<<<numBlocks, blockSize>>>(d_queue[gpu], d_values[gpu], MAX_QUEUE_SIZE);
        checkCudaError(hipGetLastError(), "insertIntoPriorityQueue");
        checkCudaError(hipDeviceSynchronize(), "hipDeviceSynchronize");

        // Extract values from the priority queue
        extractFromPriorityQueue<<<numBlocks, blockSize>>>(d_queue[gpu], d_result[gpu], MAX_QUEUE_SIZE);
        checkCudaError(hipGetLastError(), "extractFromPriorityQueue");
        checkCudaError(hipDeviceSynchronize(), "hipDeviceSynchronize");

        // Copy result back to host
        checkCudaError(hipMemcpy(h_result, d_result[gpu], MAX_QUEUE_SIZE * sizeof(int), hipMemcpyDeviceToHost), "hipMemcpy d_result");

        // Free device memory
        checkCudaError(hipFree(d_queue[gpu]), "hipFree d_queue");
        checkCudaError(hipFree(d_values[gpu]), "hipFree d_values");
        checkCudaError(hipFree(d_result[gpu]), "hipFree d_result");
    }

    // Print the result
    std::cout << "Extracted elements from priority queue:" << std::endl;
    for (int i = 0; i < MAX_QUEUE_SIZE; i++) {
        std::cout << h_result[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}