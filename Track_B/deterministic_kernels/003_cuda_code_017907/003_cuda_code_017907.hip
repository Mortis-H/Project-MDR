#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define NUM_PARAMETERS 128
#define BLOCK_SIZE 256
#define NUM_BLOCKS ((NUM_PARAMETERS + BLOCK_SIZE - 1) / BLOCK_SIZE)

__global__ void initializeParameters(float *params, hiprandState *states, int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_PARAMETERS) {
        hiprand_init(seed, idx, 0, &states[idx]);
        params[idx] = hiprand_uniform(&states[idx]) * 2.0f - 1.0f; // Initialize between -1 and 1
    }
}

__global__ void computeGradient(float *params, float *gradients) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_PARAMETERS) {
        // Example gradient: simple quadratic function gradient
        gradients[idx] = 2.0f * params[idx];
    }
}

__global__ void updateParameters(float *params, float *gradients, float learningRate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_PARAMETERS) {
        params[idx] -= learningRate * gradients[idx];
    }
}

int main() {
    float *params, *gradients;
    hiprandState *devStates;
    float learningRate = 0.01f;
    int numIterations = 1000;

    // Allocate memory on the device
    hipMalloc((void**)&params, NUM_PARAMETERS * sizeof(float));
    hipMalloc((void**)&gradients, NUM_PARAMETERS * sizeof(float));
    hipMalloc((void**)&devStates, NUM_PARAMETERS * sizeof(hiprandState));

    // Initialize parameters
    initializeParameters<<<NUM_BLOCKS, BLOCK_SIZE>>>(params, devStates, 12345);
    hipDeviceSynchronize();

    // Gradient descent loop
    for (int iter = 0; iter < numIterations; ++iter) {
        computeGradient<<<NUM_BLOCKS, BLOCK_SIZE>>>(params, gradients);
        updateParameters<<<NUM_BLOCKS, BLOCK_SIZE>>>(params, gradients, learningRate);
        hipDeviceSynchronize();
    }

    // Copy results back to host
    float *hostParams = new float[NUM_PARAMETERS];
    hipMemcpy(hostParams, params, NUM_PARAMETERS * sizeof(float), hipMemcpyDeviceToHost);

    // Output final parameters
    std::cout << "Final parameters:\n";
    for (int i = 0; i < NUM_PARAMETERS; ++i) {
        std::cout << hostParams[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    hipFree(params);
    hipFree(gradients);
    hipFree(devStates);

    // Free host memory
    delete[] hostParams;

    return 0;
}