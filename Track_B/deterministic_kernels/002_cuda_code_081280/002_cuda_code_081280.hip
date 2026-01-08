#include <iostream>
#include <hip/hip_runtime.h>


// Define a simple structure for the priority queue element
struct PQElement {
    int key;
    int value;
};

// CUDA kernel to insert elements into a priority queue
__global__ void insertIntoPQ(PQElement *pq, int *insertKeys, int *insertValues, int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        PQElement element = {insertKeys[idx], insertValues[idx]};
        int i = idx;
        // Perform heap insert operation
        while (i > 0 && pq[(i - 1) / 2].key < element.key) {
            pq[i] = pq[(i - 1) / 2];
            i = (i - 1) / 2;
        }
        pq[i] = element;
    }
}

// CUDA kernel to extract the maximum element from the priority queue
__global__ void extractMaxFromPQ(PQElement *pq, int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        // Extract max element
        PQElement maxElement = pq[0];
        // Move last element to root
        pq[0] = pq[numElements - 1];
        int i = 0;
        int left, right, largest;
        while (true) {
            left = 2 * i + 1;
            right = 2 * i + 2;
            largest = i;
            if (left < numElements - 1 && pq[left].key > pq[largest].key) {
                largest = left;
            }
            if (right < numElements - 1 && pq[right].key > pq[largest].key) {
                largest = right;
            }
            if (largest != i) {
                PQElement temp = pq[i];
                pq[i] = pq[largest];
                pq[largest] = temp;
                i = largest;
            } else {
                break;
            }
        }
    }
}

int main() {
    const int NUM_ELEMENTS = 32;
    PQElement *h_pq = new PQElement[NUM_ELEMENTS];
    int *h_insertKeys = new int[NUM_ELEMENTS];
    int *h_insertValues = new int[NUM_ELEMENTS];

    // Initialize priority queue elements with dummy values
    for (int i = 0; i < NUM_ELEMENTS; ++i) {
        h_pq[i] = {0, 0};
        h_insertKeys[i] = NUM_ELEMENTS - i; // Random keys for demonstration
        h_insertValues[i] = i; // Random values for demonstration
    }

    PQElement *d_pq;
    int *d_insertKeys, *d_insertValues;
    hipError_t err;

    // Allocate memory on the device
    err = hipMalloc((void **)&d_pq, NUM_ELEMENTS * sizeof(PQElement));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device PQ: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc((void **)&d_insertKeys, NUM_ELEMENTS * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device keys: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc((void **)&d_insertValues, NUM_ELEMENTS * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device values: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy data from host to device
    err = hipMemcpy(d_pq, h_pq, NUM_ELEMENTS * sizeof(PQElement), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy PQ to device: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMemcpy(d_insertKeys, h_insertKeys, NUM_ELEMENTS * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy keys to device: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMemcpy(d_insertValues, h_insertValues, NUM_ELEMENTS * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy values to device: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Launch kernel to insert elements into the priority queue
    dim3 blockSize(32);
    dim3 gridSize((NUM_ELEMENTS + blockSize.x - 1) / blockSize.x);
    insertIntoPQ<<<gridSize, blockSize>>>(d_pq, d_insertKeys, d_insertValues, NUM_ELEMENTS);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch insert kernel: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Launch kernel to extract the maximum element from the priority queue
    extractMaxFromPQ<<<1, 1>>>(d_pq, NUM_ELEMENTS);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch extract kernel: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy data back from device to host
    err = hipMemcpy(h_pq, d_pq, NUM_ELEMENTS * sizeof(PQElement), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy PQ back to host: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Free device memory
    hipFree(d_pq);
    hipFree(d_insertKeys);
    hipFree(d_insertValues);

    // Print the first element of the priority queue (should be the max element after extraction)
    std::cout << "Max element key: " << h_pq[0].key << ", value: " << h_pq[0].value << std::endl;

    // Clean up host memory
    delete[] h_pq;
    delete[] h_insertKeys;
    delete[] h_insertValues;

    return 0;
}