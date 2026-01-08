#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define NUM_PARAMETERS 512
#define BLOCK_SIZE 256

__global__ void initializeParameters(float* parameters, float* gradients) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_PARAMETERS) {
        // Initialize parameters and gradients
        parameters[idx] = 1.0f; // Starting point
        gradients[idx] = 0.0f;  // Initial gradient
    }
}

__global__ void computeGradients(float* parameters, float* gradients) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_PARAMETERS) {
        // Example gradient computation: gradient = 2 * parameter (simple quadratic function)
        gradients[idx] = 2.0f * parameters[idx];
    }
}

__global__ void updateParameters(float* parameters, float* gradients, float learningRate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_PARAMETERS) {
        // Update parameters using gradient descent
        parameters[idx] -= learningRate * gradients[idx];
    }
}

int main() {
    float* d_parameters;
    float* d_gradients;
    float learningRate = 0.01f;
    int numBlocks = (NUM_PARAMETERS + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Allocate memory on the device
    hipError_t err = hipMalloc((void**)&d_parameters, NUM_PARAMETERS * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for parameters: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc((void**)&d_gradients, NUM_PARAMETERS * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for gradients: " << hipGetErrorString(err) << std::endl;
        hipFree(d_parameters);
        return -1;
    }

    // Initialize parameters and gradients on the device
    initializeParameters<<<numBlocks, BLOCK_SIZE>>>(d_parameters, d_gradients);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to initialize parameters and gradients: " << hipGetErrorString(err) << std::endl;
        hipFree(d_parameters);
        hipFree(d_gradients);
        return -1;
    }
    hipDeviceSynchronize();

    // Perform a few iterations of gradient descent
    for (int i = 0; i < 100; ++i) {
        // Compute gradients
        computeGradients<<<numBlocks, BLOCK_SIZE>>>(d_parameters, d_gradients);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Failed to compute gradients: " << hipGetErrorString(err) << std::endl;
            hipFree(d_parameters);
            hipFree(d_gradients);
            return -1;
        }

        // Update parameters
        updateParameters<<<numBlocks, BLOCK_SIZE>>>(d_parameters, d_gradients, learningRate);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Failed to update parameters: " << hipGetErrorString(err) << std::endl;
            hipFree(d_parameters);
            hipFree(d_gradients);
            return -1;
        }

        // Synchronize to ensure all operations are complete
        hipDeviceSynchronize();
    }

    // Free device memory
    hipFree(d_parameters);
    hipFree(d_gradients);

    std::cout << "Gradient descent completed successfully." << std::endl;
    return 0;
}