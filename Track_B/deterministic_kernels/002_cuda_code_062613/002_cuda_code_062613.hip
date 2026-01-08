#include <iostream>
#include <hip/hip_runtime.h>


// Define constants
#define NUM_VARIABLES 256
#define BLOCK_SIZE 16

// Kernel to perform RMSprop update
__global__ void rmspropUpdate(float* grad, float* var, float* cache, float learning_rate, float decay_rate, float epsilon) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_VARIABLES) {
        // Update cache
        cache[idx] = decay_rate * cache[idx] + (1 - decay_rate) * grad[idx] * grad[idx];
        // Update variable
        var[idx] -= learning_rate * grad[idx] / (sqrtf(cache[idx]) + epsilon);
    }
}

// Kernel to initialize variables, gradients, and cache
__global__ void initialize(float* var, float* grad, float* cache) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_VARIABLES) {
        var[idx] = 1.0f;   // Initialize variables to 1.0
        grad[idx] = 0.1f;  // Initialize gradients to 0.1
        cache[idx] = 0.0f; // Initialize cache to 0.0
    }
}

int main() {
    // Device pointers
    float* d_var = nullptr;
    float* d_grad = nullptr;
    float* d_cache = nullptr;

    // Allocate memory on the device
    hipError_t err = hipMalloc((void**)&d_var, NUM_VARIABLES * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate memory for variables" << std::endl;
        return -1;
    }
    err = hipMalloc((void**)&d_grad, NUM_VARIABLES * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate memory for gradients" << std::endl;
        hipFree(d_var);
        return -1;
    }
    err = hipMalloc((void**)&d_cache, NUM_VARIABLES * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate memory for cache" << std::endl;
        hipFree(d_var);
        hipFree(d_grad);
        return -1;
    }

    // Initialize variables, gradients, and cache
    initialize<<<(NUM_VARIABLES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_var, d_grad, d_cache);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch initialize kernel" << std::endl;
        hipFree(d_var);
        hipFree(d_grad);
        hipFree(d_cache);
        return -1;
    }
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        std::cerr << "Failed to synchronize after initialize kernel" << std::endl;
        hipFree(d_var);
        hipFree(d_grad);
        hipFree(d_cache);
        return -1;
    }

    // RMSprop parameters
    float learning_rate = 0.001f;
    float decay_rate = 0.9f;
    float epsilon = 1e-8f;

    // Launch RMSprop update kernel
    rmspropUpdate<<<(NUM_VARIABLES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_grad, d_var, d_cache, learning_rate, decay_rate, epsilon);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch rmspropUpdate kernel" << std::endl;
        hipFree(d_var);
        hipFree(d_grad);
        hipFree(d_cache);
        return -1;
    }
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        std::cerr << "Failed to synchronize after rmspropUpdate kernel" << std::endl;
        hipFree(d_var);
        hipFree(d_grad);
        hipFree(d_cache);
        return -1;
    }

    // Free device memory
    hipFree(d_var);
    hipFree(d_grad);
    hipFree(d_cache);

    std::cout << "RMSprop update completed successfully." << std::endl;
    return 0;
}