#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define NUM_PARAMETERS 4096
#define BLOCK_SIZE 256

// Kernel to perform gradient descent step
__global__ void gradientDescentKernel(float* parameters, float* gradients, float learningRate, int numParameters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numParameters) {
        parameters[idx] -= learningRate * gradients[idx];
    }
}

// Kernel to initialize parameters and gradients with random values
__global__ void initializeKernel(float* parameters, float* gradients, int numParameters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numParameters) {
        hiprandState state;
        hiprand_init((unsigned long long)clock() + idx, 0, 0, &state);
        parameters[idx] = hiprand_uniform(&state) * 10.0f; // Random values between 0 and 10
        gradients[idx] = hiprand_uniform(&state) * 2.0f - 1.0f; // Random values between -1 and 1
    }
}

int main() {
    float* d_parameters, *d_gradients;
    float learningRate = 0.01f;

    // Allocate memory on the device
    hipError_t err = hipMalloc((void**)&d_parameters, NUM_PARAMETERS * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate memory for parameters: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    err = hipMalloc((void**)&d_gradients, NUM_PARAMETERS * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate memory for gradients: " << hipGetErrorString(err) << std::endl;
        hipFree(d_parameters);
        return -1;
    }

    // Initialize parameters and gradients
    int numBlocks = (NUM_PARAMETERS + BLOCK_SIZE - 1) / BLOCK_SIZE;
    initializeKernel<<<numBlocks, BLOCK_SIZE>>>(d_parameters, d_gradients, NUM_PARAMETERS);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel launch failed for initialization: " << hipGetErrorString(err) << std::endl;
        hipFree(d_parameters);
        hipFree(d_gradients);
        return -1;
    }

    // Perform a single gradient descent step
    gradientDescentKernel<<<numBlocks, BLOCK_SIZE>>>(d_parameters, d_gradients, learningRate, NUM_PARAMETERS);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel launch failed for gradient descent: " << hipGetErrorString(err) << std::endl;
        hipFree(d_parameters);
        hipFree(d_gradients);
        return -1;
    }

    // Free device memory
    hipFree(d_parameters);
    hipFree(d_gradients);

    return 0;
}