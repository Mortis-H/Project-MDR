#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

// Define the number of parameters
#define NUM_PARAMETERS 2048

// Kernel to perform gradient descent step
__global__ void gradientDescentKernel(float* parameters, float* gradients, float learningRate, int numParameters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numParameters) {
        // Perform the gradient descent update
        parameters[idx] -= learningRate * gradients[idx];
    }
}

// Kernel to initialize parameters and gradients with random values
__global__ void initializeKernel(float* parameters, float* gradients, int numParameters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numParameters) {
        hiprandState state;
        hiprand_init((unsigned long long) clock() + idx, 0, 0, &state);
        parameters[idx] = hiprand_uniform(&state) * 10.0f; // Random value between 0 and 10
        gradients[idx] = hiprand_uniform(&state) * 2.0f - 1.0f; // Random value between -1 and 1
    }
}

// Function to check for CUDA errors
void checkCudaError(hipError_t err, const char* msg) {
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << msg << " (" << hipGetErrorString(err) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    const int numParameters = NUM_PARAMETERS;
    const int blockSize = 256;
    const int numBlocks = (numParameters + blockSize - 1) / blockSize;
    const float learningRate = 0.01f;

    float* d_parameters;
    float* d_gradients;

    // Allocate memory on the device
    checkCudaError(hipMalloc((void**)&d_parameters, numParameters * sizeof(float)), "Failed to allocate device memory for parameters");
    checkCudaError(hipMalloc((void**)&d_gradients, numParameters * sizeof(float)), "Failed to allocate device memory for gradients");

    // Initialize parameters and gradients
    initializeKernel<<<numBlocks, blockSize>>>(d_parameters, d_gradients, numParameters);
    checkCudaError(hipGetLastError(), "Kernel launch failed for initialization");
    checkCudaError(hipDeviceSynchronize(), "Failed to synchronize after initialization");

    // Perform a single gradient descent step
    gradientDescentKernel<<<numBlocks, blockSize>>>(d_parameters, d_gradients, learningRate, numParameters);
    checkCudaError(hipGetLastError(), "Kernel launch failed for gradient descent");
    checkCudaError(hipDeviceSynchronize(), "Failed to synchronize after gradient descent");

    // Free device memory
    checkCudaError(hipFree(d_parameters), "Failed to free device memory for parameters");
    checkCudaError(hipFree(d_gradients), "Failed to free device memory for gradients");

    std::cout << "Gradient descent execution completed successfully." << std::endl;

    return 0;
}