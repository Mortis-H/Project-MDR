#include <iostream>
#include <hip/hip_runtime.h>

// Define the size of the priority queue
#define PQ_SIZE 1024

// Kernel to insert elements into the priority queue
__global__ void insertIntoPQ(int* pq, int* values) {
    extern __shared__ int sdata[];

    // Load data into shared memory
    int tid = threadIdx.x;
    sdata[tid] = values[tid];
    __syncthreads();

    // Perform warp-level insertion sort
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int val = sdata[tid];
        for (int offset = stride; offset > 0; offset /= 2) {
            int pos = (tid / offset) % 2;
            int other = sdata[tid - offset * (1 - pos)];
            if ((pos == 0 && val > other) || (pos == 1 && val < other)) {
                val = other;
            }
            __syncthreads();
            sdata[tid - offset * pos] = val;
            __syncthreads();
        }
    }

    // Write the sorted values back to global memory
    pq[tid] = sdata[tid];
}

// Kernel to extract the maximum element from the priority queue
__global__ void extractMaxFromPQ(int* pq, int* maxVal) {
    extern __shared__ int sdata[];

    // Load data into shared memory
    int tid = threadIdx.x;
    sdata[tid] = pq[tid];
    __syncthreads();

    // Perform warp-level reduction to find the maximum
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sdata[tid] = max(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }

    // Write the maximum value back to global memory
    if (tid == 0) {
        *maxVal = sdata[0];
    }
}

int main() {
    int* h_values = new int[PQ_SIZE];
    int* h_pq = new int[PQ_SIZE];
    int* h_maxVal = new int[1];

    // Initialize values with random numbers
    for (int i = 0; i < PQ_SIZE; i++) {
        h_values[i] = rand() % 10000;
    }

    int* d_values, *d_pq, *d_maxVal;
    size_t size = PQ_SIZE * sizeof(int);
    size_t maxValSize = sizeof(int);

    // Allocate memory on the device
    hipError_t err = hipMalloc((void**)&d_values, size);
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for values" << std::endl;
        return -1;
    }
    err = hipMalloc((void**)&d_pq, size);
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for pq" << std::endl;
        return -1;
    }
    err = hipMalloc((void**)&d_maxVal, maxValSize);
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for maxVal" << std::endl;
        return -1;
    }

    // Copy values to device
    err = hipMemcpy(d_values, h_values, size, hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy values to device" << std::endl;
        return -1;
    }

    // Insert elements into the priority queue
    insertIntoPQ<<<1, PQ_SIZE, PQ_SIZE * sizeof(int)>>>(d_pq, d_values);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel launch failed: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        std::cerr << "Failed to synchronize device: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Extract the maximum element from the priority queue
    extractMaxFromPQ<<<1, PQ_SIZE, PQ_SIZE * sizeof(int)>>>(d_pq, d_maxVal);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel launch failed: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        std::cerr << "Failed to synchronize device: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy the result back to the host
    err = hipMemcpy(h_maxVal, d_maxVal, maxValSize, hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy maxVal to host" << std::endl;
        return -1;
    }

    // Print the maximum value
    std::cout << "Maximum value in the priority queue: " << h_maxVal[0] << std::endl;

    // Free device memory
    hipFree(d_values);
    hipFree(d_pq);
    hipFree(d_maxVal);

    // Free host memory
    delete[] h_values;
    delete[] h_pq;
    delete[] h_maxVal;

    return 0;
}