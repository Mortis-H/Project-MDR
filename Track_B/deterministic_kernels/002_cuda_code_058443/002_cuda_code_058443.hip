#include <iostream>
#include <hip/hip_runtime.h>


#define MAX_QUEUE_SIZE 64

// Function to check for CUDA errors
#define cudaCheckError(err) (HandleError(err, __FILE__, __LINE__))

static void HandleError(hipError_t err, const char *file, int line) {
    if (err != hipSuccess) {
        printf("%s in %s at line %d\n", hipGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

// Kernel to insert an element into the priority queue
__global__ void insertIntoQueue(int* queue, int* size, int value) {
    int idx = atomicAdd(size, 1);
    if (idx >= MAX_QUEUE_SIZE) {
        atomicSub(size, 1); // Decrement size if queue is full
        return;
    }
    queue[idx] = value;
    // Bubble up to maintain heap property
    while (idx > 0) {
        int parentIdx = (idx - 1) / 2;
        if (queue[parentIdx] < queue[idx]) {
            int temp = queue[parentIdx];
            queue[parentIdx] = queue[idx];
            queue[idx] = temp;
            idx = parentIdx;
        } else {
            break;
        }
    }
}

// Kernel to extract the maximum element from the priority queue
__global__ void extractMaxFromQueue(int* queue, int* size) {
    int idx = atomicSub(size, 1) - 1;
    if (idx < 0) {
        atomicAdd(size, 1); // Increment size if queue is empty
        return;
    }
    queue[0] = queue[idx];
    // Heapify down to maintain heap property
    idx = 0;
    while (true) {
        int leftChildIdx = 2 * idx + 1;
        int rightChildIdx = 2 * idx + 2;
        int largestIdx = idx;
        if (leftChildIdx < *size && queue[leftChildIdx] > queue[largestIdx]) {
            largestIdx = leftChildIdx;
        }
        if (rightChildIdx < *size && queue[rightChildIdx] > queue[largestIdx]) {
            largestIdx = rightChildIdx;
        }
        if (largestIdx != idx) {
            int temp = queue[idx];
            queue[idx] = queue[largestIdx];
            queue[largestIdx] = temp;
            idx = largestIdx;
        } else {
            break;
        }
    }
}

int main() {
    int* d_queue;
    int* d_size;
    int h_size = 0;

    // Allocate memory on the device
    cudaCheckError(hipMalloc((void**)&d_queue, MAX_QUEUE_SIZE * sizeof(int)));
    cudaCheckError(hipMalloc((void**)&d_size, sizeof(int)));

    // Initialize size to 0
    cudaCheckError(hipMemcpy(d_size, &h_size, sizeof(int), hipMemcpyHostToDevice));

    // Insert elements into the priority queue
    for (int i = 0; i < MAX_QUEUE_SIZE; ++i) {
        int value = i;
        insertIntoQueue<<<1, 1>>>(d_queue, d_size, value);
        cudaCheckError(hipGetLastError());
        cudaCheckError(hipDeviceSynchronize());
    }

    // Extract elements from the priority queue
    for (int i = 0; i < MAX_QUEUE_SIZE; ++i) {
        extractMaxFromQueue<<<1, 1>>>(d_queue, d_size);
        cudaCheckError(hipGetLastError());
        cudaCheckError(hipDeviceSynchronize());
    }

    // Free device memory
    cudaCheckError(hipFree(d_queue));
    cudaCheckError(hipFree(d_size));

    std::cout << "Priority queue operations completed successfully." << std::endl;

    return 0;
}