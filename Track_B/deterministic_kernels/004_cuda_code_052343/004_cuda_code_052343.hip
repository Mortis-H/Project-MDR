#include <iostream>
#include <hip/hip_runtime.h>


#define NUM_GPUS 2
#define PQ_SIZE 64

__device__ int pq[PQ_SIZE];
__device__ int pq_size;

__global__ void initializePQ() {
    int idx = threadIdx.x;
    if (idx == 0) {
        pq_size = 0;
        for (int i = 0; i < PQ_SIZE; i++) {
            pq[i] = 0;
        }
    }
}

__global__ void insertIntoPQ(int value) {
    int idx = threadIdx.x;
    if (idx == 0) {
        if (pq_size < PQ_SIZE) {
            int pos = pq_size;
            pq[pos] = value;
            while (pos != 0 && pq[(pos - 1) / 2] < pq[pos]) {
                int temp = pq[pos];
                pq[pos] = pq[(pos - 1) / 2];
                pq[(pos - 1) / 2] = temp;
                pos = (pos - 1) / 2;
            }
            pq_size++;
        }
    }
}

__global__ void removeFromPQ() {
    int idx = threadIdx.x;
    if (idx == 0) {
        if (pq_size > 0) {
            pq[0] = pq[pq_size - 1];
            pq_size--;
            int pos = 0;
            while (true) {
                int left = 2 * pos + 1;
                int right = 2 * pos + 2;
                int largest = pos;
                if (left < pq_size && pq[left] > pq[largest]) {
                    largest = left;
                }
                if (right < pq_size && pq[right] > pq[largest]) {
                    largest = right;
                }
                if (largest != pos) {
                    int temp = pq[pos];
                    pq[pos] = pq[largest];
                    pq[largest] = temp;
                    pos = largest;
                } else {
                    break;
                }
            }
        }
    }
}

__global__ void printPQ() {
    int idx = threadIdx.x;
    if (idx == 0) {
        printf("Priority Queue: ");
        for (int i = 0; i < pq_size; i++) {
            printf("%d ", pq[i]);
        }
        printf("\n");
    }
}

int main() {
    int deviceCount;
    hipGetDeviceCount(&deviceCount);
    if (deviceCount < NUM_GPUS) {
        std::cerr << "Error: This code requires at least " << NUM_GPUS << " GPUs, but only " << deviceCount << " are available." << std::endl;
        return -1;
    }

    // Initialize priority queue on each GPU
    for (int i = 0; i < NUM_GPUS; i++) {
        hipSetDevice(i);
        initializePQ<<<1, 1>>>();
        hipDeviceSynchronize();
        hipError_t err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "CUDA error in initializePQ: " << hipGetErrorString(err) << std::endl;
            return -1;
        }
    }

    // Insert elements into the priority queue on each GPU
    for (int i = 0; i < NUM_GPUS; i++) {
        hipSetDevice(i);
        for (int j = 0; j < PQ_SIZE / NUM_GPUS; j++) {
            insertIntoPQ<<<1, 1>>>(j + i * (PQ_SIZE / NUM_GPUS));
            hipDeviceSynchronize();
            hipError_t err = hipGetLastError();
            if (err != hipSuccess) {
                std::cerr << "CUDA error in insertIntoPQ: " << hipGetErrorString(err) << std::endl;
                return -1;
            }
        }
    }

    // Print priority queue from each GPU
    for (int i = 0; i < NUM_GPUS; i++) {
        hipSetDevice(i);
        printPQ<<<1, 1>>>();
        hipDeviceSynchronize();
        hipError_t err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "CUDA error in printPQ: " << hipGetErrorString(err) << std::endl;
            return -1;
        }
    }

    // Remove elements from the priority queue on each GPU
    for (int i = 0; i < NUM_GPUS; i++) {
        hipSetDevice(i);
        for (int j = 0; j < PQ_SIZE / NUM_GPUS; j++) {
            removeFromPQ<<<1, 1>>>();
            hipDeviceSynchronize();
            hipError_t err = hipGetLastError();
            if (err != hipSuccess) {
                std::cerr << "CUDA error in removeFromPQ: " << hipGetErrorString(err) << std::endl;
                return -1;
            }
        }
    }

    // Print priority queue from each GPU after removal
    for (int i = 0; i < NUM_GPUS; i++) {
        hipSetDevice(i);
        printPQ<<<1, 1>>>();
        hipDeviceSynchronize();
        hipError_t err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "CUDA error in printPQ: " << hipGetErrorString(err) << std::endl;
            return -1;
        }
    }

    return 0;
}