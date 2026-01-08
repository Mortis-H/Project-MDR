#include <iostream>
#include <hip/hip_runtime.h>


#define MAX_QUEUE_SIZE 128
#define NUM_GPUS 2

// Define a simple node structure for the priority queue
struct Node {
    int priority;
    int value;
};

// Kernel to initialize the priority queue on the device
__global__ void initializeQueue(Node* d_queue, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_queue[idx].priority = 0;
        d_queue[idx].value = 0;
    }
}

// Kernel to insert an element into the priority queue
__global__ void insertElement(Node* d_queue, int size, int priority, int value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Simple linear search to find the correct position
        for (int i = 0; i < size; ++i) {
            if (d_queue[i].priority < priority) {
                // Shift elements to the right
                for (int j = size - 1; j > i; --j) {
                    d_queue[j] = d_queue[j - 1];
                }
                d_queue[i].priority = priority;
                d_queue[i].value = value;
                break;
            }
        }
    }
}

// Kernel to extract the maximum element from the priority queue
__global__ void extractMax(Node* d_queue, int size, int* d_maxValue) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        *d_maxValue = d_queue[0].value;
        // Shift elements to the left
        for (int i = 0; i < size - 1; ++i) {
            d_queue[i] = d_queue[i + 1];
        }
        d_queue[size - 1].priority = 0;
        d_queue[size - 1].value = 0;
    }
}

// Host function to check for CUDA errors
void checkCudaError(hipError_t err, const char* msg) {
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << msg << " - " << hipGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    int numDevices;
    hipGetDeviceCount(&numDevices);
    if (numDevices < NUM_GPUS) {
        std::cerr << "Error: Not enough GPUs available. Required: " << NUM_GPUS << ", Available: " << numDevices << std::endl;
        return EXIT_FAILURE;
    }

    Node* h_queues[NUM_GPUS];
    Node* d_queues[NUM_GPUS];
    int* d_maxValues[NUM_GPUS];

    int blockSize = 256;
    int numBlocks = (MAX_QUEUE_SIZE + blockSize - 1) / blockSize;

    for (int gpu = 0; gpu < NUM_GPUS; ++gpu) {
        hipSetDevice(gpu);

        // Allocate memory on the host and device
        h_queues[gpu] = (Node*)malloc(MAX_QUEUE_SIZE * sizeof(Node));
        checkCudaError(hipMalloc((void**)&d_queues[gpu], MAX_QUEUE_SIZE * sizeof(Node)), "hipMalloc d_queues");
        checkCudaError(hipMalloc((void**)&d_maxValues[gpu], sizeof(int)), "hipMalloc d_maxValues");

        // Initialize the priority queue on the device
        initializeQueue<<<numBlocks, blockSize>>>(d_queues[gpu], MAX_QUEUE_SIZE);
        checkCudaError(hipGetLastError(), "initializeQueue kernel launch");
        checkCudaError(hipDeviceSynchronize(), "hipDeviceSynchronize after initializeQueue");

        // Insert elements into the priority queue
        for (int i = 0; i < MAX_QUEUE_SIZE; ++i) {
            insertElement<<<numBlocks, blockSize>>>(d_queues[gpu], MAX_QUEUE_SIZE, i, i * 10);
            checkCudaError(hipGetLastError(), "insertElement kernel launch");
            checkCudaError(hipDeviceSynchronize(), "hipDeviceSynchronize after insertElement");
        }

        // Extract the maximum element from the priority queue
        extractMax<<<1, 1>>>(d_queues[gpu], MAX_QUEUE_SIZE, d_maxValues[gpu]);
        checkCudaError(hipGetLastError(), "extractMax kernel launch");
        checkCudaError(hipDeviceSynchronize(), "hipDeviceSynchronize after extractMax");

        // Copy the result back to the host
        int maxValue;
        checkCudaError(hipMemcpy(&maxValue, d_maxValues[gpu], sizeof(int), hipMemcpyDeviceToHost), "hipMemcpy d_maxValues");

        // Print the maximum value extracted from the priority queue
        std::cout << "GPU " << gpu << ": Max value extracted from priority queue: " << maxValue << std::endl;

        // Free device memory
        checkCudaError(hipFree(d_queues[gpu]), "hipFree d_queues");
        checkCudaError(hipFree(d_maxValues[gpu]), "hipFree d_maxValues");

        // Free host memory
        free(h_queues[gpu]);
    }

    return EXIT_SUCCESS;
}