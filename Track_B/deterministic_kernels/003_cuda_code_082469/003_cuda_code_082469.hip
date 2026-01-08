#include <iostream>
#include <hip/hip_runtime.h>
#include <vector>
#include <hiprand/hiprand_kernel.h>

// Define the number of variables and the number of GPUs
#define NUM_VARIABLES 1024
#define NUM_GPUS 2

// Kernel to initialize random values for variables and gradients
__global__ void initialize(float *variables, float *gradients, hiprandState *states, int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_VARIABLES) {
        hiprand_init(seed, idx, 0, &states[idx]);
        variables[idx] = hiprand_uniform(&states[idx]) * 10.0f;
        gradients[idx] = 0.0f;
    }
}

// Kernel to compute the gradient (simple example: gradient of x^2 is 2x)
__global__ void computeGradient(float *variables, float *gradients) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_VARIABLES) {
        gradients[idx] = 2.0f * variables[idx];
    }
}

// Kernel to update variables using gradient descent
__global__ void updateVariables(float *variables, float *gradients, float learningRate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_VARIABLES) {
        variables[idx] -= learningRate * gradients[idx];
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
    float *variables[NUM_GPUS], *gradients[NUM_GPUS];
    hiprandState *states[NUM_GPUS];
    hipStream_t streams[NUM_GPUS];
    int blockSize = 256;
    int numBlocks = (NUM_VARIABLES + blockSize - 1) / blockSize;
    float learningRate = 0.01f;
    float *h_variables = new float[NUM_VARIABLES];

    // Allocate and initialize data for each GPU
    for (int i = 0; i < NUM_GPUS; ++i) {
        hipSetDevice(i);
        hipStreamCreate(&streams[i]);
        hipMalloc(&variables[i], NUM_VARIABLES * sizeof(float));
        hipMalloc(&gradients[i], NUM_VARIABLES * sizeof(float));
        hipMalloc(&states[i], NUM_VARIABLES * sizeof(hiprandState));
        initialize<<<numBlocks, blockSize, 0, streams[i]>>>(variables[i], gradients[i], states[i], i);
        checkCudaError(hipGetLastError(), "Kernel launch failed");
        checkCudaError(hipDeviceSynchronize(), "Device synchronization failed");
    }

    // Perform gradient descent
    for (int iter = 0; iter < 1000; ++iter) {
        for (int i = 0; i < NUM_GPUS; ++i) {
            hipSetDevice(i);
            computeGradient<<<numBlocks, blockSize, 0, streams[i]>>>(variables[i], gradients[i]);
            checkCudaError(hipGetLastError(), "Kernel launch failed");
            updateVariables<<<numBlocks, blockSize, 0, streams[i]>>>(variables[i], gradients[i], learningRate);
            checkCudaError(hipGetLastError(), "Kernel launch failed");
        }
        // Synchronize all streams
        for (int i = 0; i < NUM_GPUS; ++i) {
            hipSetDevice(i);
            checkCudaError(hipDeviceSynchronize(), "Device synchronization failed");
        }
    }

    // Copy results back to host
    hipSetDevice(0);
    hipMemcpy(h_variables, variables[0], NUM_VARIABLES * sizeof(float), hipMemcpyDeviceToHost);
    checkCudaError(hipGetLastError(), "Memory copy failed");

    // Clean up
    for (int i = 0; i < NUM_GPUS; ++i) {
        hipSetDevice(i);
        hipFree(variables[i]);
        hipFree(gradients[i]);
        hipFree(states[i]);
        hipStreamDestroy(streams[i]);
    }

    delete[] h_variables;

    std::cout << "Gradient descent completed successfully." << std::endl;
    return 0;
}