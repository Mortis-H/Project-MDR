#include <iostream>
#include <hip/hip_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <hiprand/hiprand_kernel.h>

#define NUM_ELEMENTS 32
#define MAX_LEVEL 4

__device__ int randomLevel(hiprandState *state) {
    int lvl = 1;
    while (hiprand_uniform(state) < 0.5 && lvl < MAX_LEVEL) lvl++;
    return lvl;
}

__global__ void initCurandStates(hiprandState *states, unsigned long seed) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < NUM_ELEMENTS) {
        hiprand_init(seed, id, 0, &states[id]);
    }
}

__global__ void skipListInsert(int *keys, int *levels, hiprandState *states) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < NUM_ELEMENTS) {
        // Generate random key and level
        keys[id] = id; // For simplicity, use id as key
        levels[id] = randomLevel(&states[id]);
    }
}

__global__ void printSkipList(int *keys, int *levels) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < NUM_ELEMENTS) {
        printf("Key: %d, Level: %d\n", keys[id], levels[id]);
    }
}

int main() {
    int *d_keys, *d_levels;
    hiprandState *d_states;

    // Allocate device memory
    hipError_t err = hipMalloc(&d_keys, NUM_ELEMENTS * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device keys memory: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc(&d_levels, NUM_ELEMENTS * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device levels memory: " << hipGetErrorString(err) << std::endl;
        hipFree(d_keys);
        return -1;
    }
    err = hipMalloc(&d_states, NUM_ELEMENTS * sizeof(hiprandState));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device states memory: " << hipGetErrorString(err) << std::endl;
        hipFree(d_keys);
        hipFree(d_levels);
        return -1;
    }

    // Initialize curand states
    initCurandStates<<<(NUM_ELEMENTS + 255) / 256, 256>>>(d_states, 123456ULL);
    hipDeviceSynchronize();

    // Asynchronously copy data to device
    hipStream_t stream;
    hipStreamCreate(&stream);

    // Insert into skip list
    skipListInsert<<<(NUM_ELEMENTS + 255) / 256, 256, 0, stream>>>(d_keys, d_levels, d_states);
    hipDeviceSynchronize();

    // Print skip list
    printSkipList<<<(NUM_ELEMENTS + 255) / 256, 256>>>(d_keys, d_levels);
    hipDeviceSynchronize();

    // Free device memory
    hipFree(d_keys);
    hipFree(d_levels);
    hipFree(d_states);
    hipStreamDestroy(stream);

    return 0;
}