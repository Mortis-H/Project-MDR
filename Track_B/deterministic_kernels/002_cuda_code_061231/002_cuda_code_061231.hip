#include <iostream>
#include <hip/hip_runtime.h>

// Define the size of the priority queue
#define PQ_SIZE 4096

// Kernel to initialize the priority queue in shared memory
__global__ void initPQ(int* d_pq, int value) {
    extern __shared__ int s_pq[];

    // Calculate the global index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize the shared memory
    if (idx < PQ_SIZE) {
        s_pq[idx] = value;
    }

    // Synchronize to ensure all elements are initialized
    __syncthreads();

    // Write the shared memory back to global memory
    if (idx < PQ_SIZE) {
        d_pq[idx] = s_pq[idx];
    }
}

// Kernel to perform a simple operation on the priority queue
__global__ void processPQ(int* d_pq) {
    extern __shared__ int s_pq[];

    // Calculate the global index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data from global memory to shared memory
    if (idx < PQ_SIZE) {
        s_pq[idx] = d_pq[idx];
    }

    // Synchronize to ensure all data is loaded
    __syncthreads();

    // Perform a simple operation (e.g., increment each element)
    if (idx < PQ_SIZE) {
        s_pq[idx] += 1;
    }

    // Synchronize to ensure all operations are complete
    __syncthreads();

    // Write the modified data back to global memory
    if (idx < PQ_SIZE) {
        d_pq[idx] = s_pq[idx];
    }
}

int main() {
    int* d_pq;
    int h_pq[PQ_SIZE];

    // Allocate memory on the device
    hipError_t err = hipMalloc((void**)&d_pq, PQ_SIZE * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Initialize the priority queue on the device
    int blockSize = 256;
    int numBlocks = (PQ_SIZE + blockSize - 1) / blockSize;
    initPQ<<<numBlocks, blockSize, PQ_SIZE * sizeof(int)>>>(d_pq, 0);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch initPQ kernel: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Synchronize to ensure initialization is complete
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        std::cerr << "Failed to synchronize after initPQ: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Process the priority queue on the device
    processPQ<<<numBlocks, blockSize, PQ_SIZE * sizeof(int)>>>(d_pq);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch processPQ kernel: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Synchronize to ensure processing is complete
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        std::cerr << "Failed to synchronize after processPQ: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy the result back to the host
    err = hipMemcpy(h_pq, d_pq, PQ_SIZE * sizeof(int), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy data from device to host: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Print the first 10 elements to verify the result
    for (int i = 0; i < 10; ++i) {
        std::cout << "Element " << i << ": " << h_pq[i] << std::endl;
    }

    // Free the device memory
    hipFree(d_pq);

    return 0;
}