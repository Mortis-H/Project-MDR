#include <iostream>
#include <hip/hip_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/extrema.h>
#include <thrust/gather.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <hiprand/hiprand_kernel.h>

#define NUM_ELEMENTS 16384
#define BLOCK_SIZE 256
#define NUM_BLOCKS (NUM_ELEMENTS / BLOCK_SIZE)

// Kernel to initialize random values in the array
__global__ void initRandom(int *d_data, int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_ELEMENTS) {
        hiprandState state;
        hiprand_init(seed, idx, 0, &state);
        d_data[idx] = hiprand(&state) % 100000;
    }
}

// Kernel to perform a simple up-sweep for building a binary heap
__global__ void upSweep(int *d_data, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ai = idx * (1 << (d + 1)) + (1 << d) - 1;
    int bi = ai + (1 << d);
    if (bi < NUM_ELEMENTS) {
        d_data[bi] += d_data[ai];
    }
}

// Kernel to perform a simple down-sweep for building a binary heap
__global__ void downSweep(int *d_data, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ai = idx * (1 << (d + 1)) + (1 << d) - 1;
    int bi = ai + (1 << d);
    if (bi < NUM_ELEMENTS) {
        int t = d_data[ai];
        d_data[ai] = d_data[bi];
        d_data[bi] += t;
    }
}

// Function to check for CUDA errors
void checkCudaError(hipError_t err, const char *msg) {
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << msg << " (" << hipGetErrorString(err) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Initialize random seed
    int seed = 12345;

    // Allocate device memory
    int *d_data;
    checkCudaError(hipMalloc(&d_data, NUM_ELEMENTS * sizeof(int)), "Failed to allocate device memory");

    // Initialize random values on the device
    initRandom<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_data, seed);
    checkCudaError(hipGetLastError(), "Kernel launch failed for initRandom");
    checkCudaError(hipDeviceSynchronize(), "Failed to synchronize after initRandom");

    // Perform up-sweep for building the binary heap
    for (int d = 0; d < log2(NUM_ELEMENTS); ++d) {
        upSweep<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_data, d);
        checkCudaError(hipGetLastError(), "Kernel launch failed for upSweep");
        checkCudaError(hipDeviceSynchronize(), "Failed to synchronize after upSweep");
    }

    // Set the last element to 0
    checkCudaError(hipMemset(&d_data[NUM_ELEMENTS - 1], 0, sizeof(int)), "Failed to set last element to 0");

    // Perform down-sweep for building the binary heap
    for (int d = log2(NUM_ELEMENTS) - 1; d >= 0; --d) {
        downSweep<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_data, d);
        checkCudaError(hipGetLastError(), "Kernel launch failed for downSweep");
        checkCudaError(hipDeviceSynchronize(), "Failed to synchronize after downSweep");
    }

    // Copy the result back to the host
    thrust::device_ptr<int> d_ptr(d_data);
    thrust::host_vector<int> h_data(d_ptr, d_ptr + NUM_ELEMENTS);

    // Print the first 10 elements of the sorted array
    std::cout << "First 10 elements of the priority queue:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << h_data[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    checkCudaError(hipFree(d_data), "Failed to free device memory");

    return 0;
}