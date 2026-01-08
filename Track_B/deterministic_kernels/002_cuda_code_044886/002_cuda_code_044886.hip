#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define NUM_ELEMENTS 4096
#define NUM_LEVELS 4
#define MAX_LEVEL 3

__constant__ int level_sizes[NUM_LEVELS] = {NUM_ELEMENTS, NUM_ELEMENTS / 2, NUM_ELEMENTS / 4, NUM_ELEMENTS / 8};

__global__ void initializeSkipList(hiprandState *states, int *skipList) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_ELEMENTS) {
        // Initialize random states
        hiprand_init((unsigned long long) clock() + idx, 0, 0, &states[idx]);
        // Randomly assign levels for each node
        int level = 0;
        while (level < MAX_LEVEL && hiprand_uniform(&states[idx]) < 0.5f) {
            level++;
        }
        // Assign level to the skip list
        skipList[idx * NUM_LEVELS] = level;
    }
}

__global__ void searchSkipList(int *skipList, int target, int *result) {
    extern __shared__ int sharedList[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory to avoid bank conflicts
    if (idx < NUM_ELEMENTS) {
        for (int l = 0; l <= MAX_LEVEL; l++) {
            sharedList[threadIdx.x * (MAX_LEVEL + 1) + l] = skipList[idx * NUM_LEVELS + l];
        }
    }
    __syncthreads();

    // Perform search
    if (idx == 0) {
        int current = 0;
        for (int level = MAX_LEVEL; level >= 0; level--) {
            while (current < NUM_ELEMENTS && sharedList[current * (MAX_LEVEL + 1) + level] >= level) {
                if (current + 1 < NUM_ELEMENTS && sharedList[(current + 1) * (MAX_LEVEL + 1)] <= target) {
                    current++;
                } else {
                    break;
                }
            }
        }
        // Check if the found element is the target
        if (current < NUM_ELEMENTS && sharedList[current * (MAX_LEVEL + 1)] == target) {
            *result = current;
        } else {
            *result = -1; // Target not found
        }
    }
}

int main() {
    int *skipList_d, *result_d;
    hiprandState *states_d;
    int *skipList_h = new int[NUM_ELEMENTS * NUM_LEVELS];
    int *result_h = new int[1];
    *result_h = -1;

    // Allocate memory on device
    hipError_t err = hipMalloc((void **)&skipList_d, NUM_ELEMENTS * NUM_LEVELS * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate skipList_d (error code " << err << ")" << std::endl;
        return -1;
    }
    err = hipMalloc((void **)&result_d, sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate result_d (error code " << err << ")" << std::endl;
        return -1;
    }
    err = hipMalloc((void **)&states_d, NUM_ELEMENTS * sizeof(hiprandState));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate states_d (error code " << err << ")" << std::endl;
        return -1;
    }

    // Initialize random states and skip list
    int blockSize = 256;
    int numBlocks = (NUM_ELEMENTS + blockSize - 1) / blockSize;
    initializeSkipList<<<numBlocks, blockSize>>>(states_d, skipList_d);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch initializeSkipList (error code " << err << ")" << std::endl;
        return -1;
    }
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        std::cerr << "Failed to synchronize device (error code " << err << ")" << std::endl;
        return -1;
    }

    // Search for a target value in the skip list
    int target = 2048; // Example target
    searchSkipList<<<1, blockSize, blockSize * (MAX_LEVEL + 1) * sizeof(int)>>>(skipList_d, target, result_d);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch searchSkipList (error code " << err << ")" << std::endl;
        return -1;
    }
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        std::cerr << "Failed to synchronize device (error code " << err << ")" << std::endl;
        return -1;
    }

    // Copy result back to host
    err = hipMemcpy(result_h, result_d, sizeof(int), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy result_d to result_h (error code " << err << ")" << std::endl;
        return -1;
    }

    // Output the result
    std::cout << "Search for target " << target << ": ";
    if (*result_h != -1) {
        std::cout << "Found at index " << *result_h << std::endl;
    } else {
        std::cout << "Not found" << std::endl;
    }

    // Free device memory
    hipFree(skipList_d);
    hipFree(result_d);
    hipFree(states_d);

    // Free host memory
    delete[] skipList_h;
    delete[] result_h;

    return 0;
}