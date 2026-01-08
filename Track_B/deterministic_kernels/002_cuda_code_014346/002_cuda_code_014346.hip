#include <iostream>
#include <hip/hip_runtime.h>

#define MAX_ELEMENTS 2048

__global__ void insertIntoPriorityQueue(int *queue, int *values, int numValues) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numValues) {
        // Atomically insert the value into the priority queue
        // We assume the queue is a max-heap
        int value = values[idx];
        int pos = atomicAdd(queue, 1);
        if (pos < MAX_ELEMENTS) {
            queue[pos + 1] = value; // Insert at the end
            // Heapify up
            while (pos > 0) {
                int parent = (pos - 1) / 2;
                if (queue[parent + 1] < queue[pos + 1]) {
                    int temp = queue[parent + 1];
                    queue[parent + 1] = queue[pos + 1];
                    queue[pos + 1] = temp;
                    pos = parent;
                } else {
                    break;
                }
            }
        }
    }
}

__global__ void extractFromPriorityQueue(int *queue, int *extractedValues, int numExtracts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numExtracts) {
        // Atomically extract the maximum value from the priority queue
        int size = atomicSub(queue, 1);
        if (size > 0) {
            extractedValues[idx] = queue[1]; // Extract the root
            queue[1] = queue[size]; // Move the last element to the root
            int pos = 0;
            // Heapify down
            while (true) {
                int left = 2 * pos + 1;
                int right = 2 * pos + 2;
                int largest = pos;
                if (left < size - 1 && queue[left + 1] > queue[largest + 1]) {
                    largest = left;
                }
                if (right < size - 1 && queue[right + 1] > queue[largest + 1]) {
                    largest = right;
                }
                if (largest != pos) {
                    int temp = queue[pos + 1];
                    queue[pos + 1] = queue[largest + 1];
                    queue[largest + 1] = temp;
                    pos = largest;
                } else {
                    break;
                }
            }
        } else {
            extractedValues[idx] = -1; // Indicate no more elements
        }
    }
}

int main() {
    int *d_queue, *d_values, *d_extractedValues;
    int h_queue[MAX_ELEMENTS + 1] = {0}; // First element is the size of the queue
    int h_values[MAX_ELEMENTS] = {30, 10, 20, 5, 15, 25, 40, 35, 45, 50}; // Example values
    int h_extractedValues[MAX_ELEMENTS] = {0};

    // Allocate device memory
    hipError_t err = hipMalloc((void**)&d_queue, (MAX_ELEMENTS + 1) * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device queue memory: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc((void**)&d_values, MAX_ELEMENTS * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device values memory: " << hipGetErrorString(err) << std::endl;
        hipFree(d_queue);
        return -1;
    }
    err = hipMalloc((void**)&d_extractedValues, MAX_ELEMENTS * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device extracted values memory: " << hipGetErrorString(err) << std::endl;
        hipFree(d_queue);
        hipFree(d_values);
        return -1;
    }

    // Copy data to device
    err = hipMemcpy(d_queue, h_queue, (MAX_ELEMENTS + 1) * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy queue to device: " << hipGetErrorString(err) << std::endl;
        hipFree(d_queue);
        hipFree(d_values);
        hipFree(d_extractedValues);
        return -1;
    }
    err = hipMemcpy(d_values, h_values, MAX_ELEMENTS * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy values to device: " << hipGetErrorString(err) << std::endl;
        hipFree(d_queue);
        hipFree(d_values);
        hipFree(d_extractedValues);
        return -1;
    }

    // Insert values into the priority queue
    int numValues = 10; // Number of values to insert
    int threadsPerBlock = 256;
    int blocksPerGrid = (numValues + threadsPerBlock - 1) / threadsPerBlock;
    insertIntoPriorityQueue<<<blocksPerGrid, threadsPerBlock>>>(d_queue, d_values, numValues);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel launch failed: " << hipGetErrorString(err) << std::endl;
        hipFree(d_queue);
        hipFree(d_values);
        hipFree(d_extractedValues);
        return -1;
    }
    hipDeviceSynchronize();

    // Extract values from the priority queue
    int numExtracts = 10; // Number of values to extract
    blocksPerGrid = (numExtracts + threadsPerBlock - 1) / threadsPerBlock;
    extractFromPriorityQueue<<<blocksPerGrid, threadsPerBlock>>>(d_queue, d_extractedValues, numExtracts);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel launch failed: " << hipGetErrorString(err) << std::endl;
        hipFree(d_queue);
        hipFree(d_values);
        hipFree(d_extractedValues);
        return -1;
    }
    hipDeviceSynchronize();

    // Copy results back to host
    err = hipMemcpy(h_extractedValues, d_extractedValues, MAX_ELEMENTS * sizeof(int), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy extracted values from device: " << hipGetErrorString(err) << std::endl;
        hipFree(d_queue);
        hipFree(d_values);
        hipFree(d_extractedValues);
        return -1;
    }

    // Print extracted values
    std::cout << "Extracted values from priority queue:" << std::endl;
    for (int i = 0; i < numExtracts; ++i) {
        std::cout << h_extractedValues[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    hipFree(d_queue);
    hipFree(d_values);
    hipFree(d_extractedValues);

    return 0;
}