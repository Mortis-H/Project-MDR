#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define NUM_PARAMS 32
#define BLOCK_SIZE 256
#define NUM_BLOCKS 128
#define NUM_ITERATIONS 1000
#define LEARNING_RATE 0.01f

__global__ void initializeParameters(float* params, hiprandState* states, int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_PARAMS) {
        hiprandState localState = states[idx];
        hiprand_init(seed, idx, 0, &localState);
        params[idx] = hiprand_uniform(&localState) * 10.0f; // Random initial values
        states[idx] = localState;
    }
}

__global__ void computeGradient(float* params, float* gradients) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_PARAMS) {
        // Example gradient computation: simple quadratic function f(x) = x^2
        // Gradient f'(x) = 2x
        gradients[idx] = 2.0f * params[idx];
    }
}

__global__ void updateParameters(float* params, float* gradients, float learningRate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_PARAMS) {
        // Update rule: x = x - learningRate * gradient
        params[idx] -= learningRate * gradients[idx];
    }
}

int main() {
    float* d_params;
    float* d_gradients;
    hiprandState* d_states;

    // Allocate memory on the device
    hipError_t err = hipMalloc(&d_params, NUM_PARAMS * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for parameters" << std::endl;
        return -1;
    }

    err = hipMalloc(&d_gradients, NUM_PARAMS * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for gradients" << std::endl;
        hipFree(d_params);
        return -1;
    }

    err = hipMalloc(&d_states, NUM_PARAMS * sizeof(hiprandState));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for states" << std::endl;
        hipFree(d_params);
        hipFree(d_gradients);
        return -1;
    }

    // Initialize parameters with random values
    initializeParameters<<<(NUM_PARAMS + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_params, d_states, 1234);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel launch failed: " << hipGetErrorString(err) << std::endl;
        hipFree(d_params);
        hipFree(d_gradients);
        hipFree(d_states);
        return -1;
    }

    // Perform gradient descent
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        computeGradient<<<(NUM_PARAMS + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_params, d_gradients);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Kernel launch failed: " << hipGetErrorString(err) << std::endl;
            hipFree(d_params);
            hipFree(d_gradients);
            hipFree(d_states);
            return -1;
        }

        updateParameters<<<(NUM_PARAMS + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_params, d_gradients, LEARNING_RATE);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Kernel launch failed: " << hipGetErrorString(err) << std::endl;
            hipFree(d_params);
            hipFree(d_gradients);
            hipFree(d_states);
            return -1;
        }
    }

    // Copy results back to host
    float h_params[NUM_PARAMS];
    err = hipMemcpy(h_params, d_params, NUM_PARAMS * sizeof(float), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy memory from device to host" << std::endl;
        hipFree(d_params);
        hipFree(d_gradients);
        hipFree(d_states);
        return -1;
    }

    // Output final parameters
    std::cout << "Final parameters:" << std::endl;
    for (int i = 0; i < NUM_PARAMS; ++i) {
        std::cout << "Param " << i << ": " << h_params[i] << std::endl;
    }

    // Free device memory
    hipFree(d_params);
    hipFree(d_gradients);
    hipFree(d_states);

    return 0;
}