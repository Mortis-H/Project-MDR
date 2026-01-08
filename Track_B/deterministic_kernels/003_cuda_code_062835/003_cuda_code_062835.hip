#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define NUM_PARAMETERS 4096
#define BLOCK_SIZE 256

__global__ void initializeParameters(float* parameters, hiprandState* states, int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_PARAMETERS) {
        hiprandState localState = states[idx];
        hiprand_init(seed, idx, 0, &localState);
        parameters[idx] = hiprand_uniform(&localState) * 2.0f - 1.0f; // Initialize to [-1, 1]
        states[idx] = localState;
    }
}

__global__ void gradientDescentStep(float* parameters, float* gradients, float learningRate, int numParameters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numParameters) {
        parameters[idx] -= learningRate * gradients[idx];
    }
}

__global__ void computeGradients(float* parameters, float* gradients, int numParameters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numParameters) {
        // Example gradient computation: simple quadratic function f(x) = x^2
        // Gradient: df/dx = 2x
        gradients[idx] = 2.0f * parameters[idx];
    }
}

void checkCudaError(hipError_t err, const char* msg) {
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << msg << " - " << hipGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    float* d_parameters;
    float* d_gradients;
    hiprandState* d_states;
    float* h_parameters = new float[NUM_PARAMETERS];
    float* h_gradients = new float[NUM_PARAMETERS];
    float learningRate = 0.01f;
    int numIterations = 1000;

    // Allocate device memory
    checkCudaError(hipMalloc(&d_parameters, NUM_PARAMETERS * sizeof(float)), "Allocating d_parameters");
    checkCudaError(hipMalloc(&d_gradients, NUM_PARAMETERS * sizeof(float)), "Allocating d_gradients");
    checkCudaError(hipMalloc(&d_states, NUM_PARAMETERS * sizeof(hiprandState)), "Allocating d_states");

    // Initialize parameters with random values
    int blocks = (NUM_PARAMETERS + BLOCK_SIZE - 1) / BLOCK_SIZE;
    initializeParameters<<<blocks, BLOCK_SIZE>>>(d_parameters, d_states, 1234);
    checkCudaError(hipGetLastError(), "Kernel launch failed for initializeParameters");

    // Perform gradient descent
    for (int i = 0; i < numIterations; ++i) {
        computeGradients<<<blocks, BLOCK_SIZE>>>(d_parameters, d_gradients, NUM_PARAMETERS);
        checkCudaError(hipGetLastError(), "Kernel launch failed for computeGradients");

        gradientDescentStep<<<blocks, BLOCK_SIZE>>>(d_parameters, d_gradients, learningRate, NUM_PARAMETERS);
        checkCudaError(hipGetLastError(), "Kernel launch failed for gradientDescentStep");
    }

    // Copy results back to host
    checkCudaError(hipMemcpy(h_parameters, d_parameters, NUM_PARAMETERS * sizeof(float), hipMemcpyDeviceToHost), "Copying parameters to host");
    checkCudaError(hipMemcpy(h_gradients, d_gradients, NUM_PARAMETERS * sizeof(float), hipMemcpyDeviceToHost), "Copying gradients to host");

    // Print first 10 parameters for verification
    std::cout << "First 10 optimized parameters:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << h_parameters[i] << std::endl;
    }

    // Free device memory
    hipFree(d_parameters);
    hipFree(d_gradients);
    hipFree(d_states);

    // Free host memory
    delete[] h_parameters;
    delete[] h_gradients;

    return 0;
}