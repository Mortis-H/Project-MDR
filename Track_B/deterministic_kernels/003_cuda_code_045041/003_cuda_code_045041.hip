#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define BLOCK_SIZE 256
#define NUM_PARAMS 8192
#define NUM_GPUS 2

__global__ void initializeParameters(float* params, float* gradients, hiprandState* states, int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_PARAMS) {
        hiprandState localState = states[idx];
        hiprand_init(seed, idx, 0, &localState);
        params[idx] = hiprand_uniform(&localState) * 2.0f - 1.0f; // Initialize between -1 and 1
        gradients[idx] = 0.0f;
        states[idx] = localState;
    }
}

__global__ void computeGradients(float* params, float* gradients, float* lossGradient) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_PARAMS) {
        // Simulate gradient computation, e.g., gradient of a simple quadratic function
        gradients[idx] = 2.0f * params[idx] + lossGradient[idx];
    }
}

__global__ void updateParameters(float* params, float* gradients, float learningRate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_PARAMS) {
        params[idx] -= learningRate * gradients[idx];
    }
}

void checkCudaError(hipError_t err, const char* msg) {
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << msg << " (" << hipGetErrorString(err) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    float* h_params = new float[NUM_PARAMS];
    float* h_gradients = new float[NUM_PARAMS];
    float* h_lossGradient = new float[NUM_PARAMS];
    float learningRate = 0.01f;

    // Initialize loss gradient with some values (example)
    for (int i = 0; i < NUM_PARAMS; ++i) {
        h_lossGradient[i] = 1.0f; // Example loss gradient
    }

    // Allocate memory on GPUs
    float* d_params[NUM_GPUS];
    float* d_gradients[NUM_GPUS];
    float* d_lossGradient[NUM_GPUS];
    hiprandState* d_states[NUM_GPUS];

    int paramsPerGPU = NUM_PARAMS / NUM_GPUS;

    for (int i = 0; i < NUM_GPUS; ++i) {
        hipSetDevice(i);
        checkCudaError(hipMalloc(&d_params[i], paramsPerGPU * sizeof(float)), "Failed to allocate params");
        checkCudaError(hipMalloc(&d_gradients[i], paramsPerGPU * sizeof(float)), "Failed to allocate gradients");
        checkCudaError(hipMalloc(&d_lossGradient[i], paramsPerGPU * sizeof(float)), "Failed to allocate lossGradient");
        checkCudaError(hipMalloc(&d_states[i], paramsPerGPU * sizeof(hiprandState)), "Failed to allocate states");

        // Copy loss gradient to device
        checkCudaError(hipMemcpy(d_lossGradient[i], h_lossGradient + i * paramsPerGPU, paramsPerGPU * sizeof(float), hipMemcpyHostToDevice), "Failed to copy lossGradient to device");

        // Initialize parameters on GPU
        dim3 blockSize(BLOCK_SIZE);
        dim3 gridSize((paramsPerGPU + blockSize.x - 1) / blockSize.x);
        initializeParameters<<<gridSize, blockSize>>>(d_params[i], d_gradients[i], d_states[i], i);
        checkCudaError(hipGetLastError(), "Kernel launch failed for initializeParameters");
    }

    // Perform gradient descent
    for (int iter = 0; iter < 100; ++iter) {
        for (int i = 0; i < NUM_GPUS; ++i) {
            hipSetDevice(i);
            dim3 blockSize(BLOCK_SIZE);
            dim3 gridSize((paramsPerGPU + blockSize.x - 1) / blockSize.x);
            computeGradients<<<gridSize, blockSize>>>(d_params[i], d_gradients[i], d_lossGradient[i]);
            checkCudaError(hipGetLastError(), "Kernel launch failed for computeGradients");

            updateParameters<<<gridSize, blockSize>>>(d_params[i], d_gradients[i], learningRate);
            checkCudaError(hipGetLastError(), "Kernel launch failed for updateParameters");
        }
    }

    // Copy parameters back to host
    for (int i = 0; i < NUM_GPUS; ++i) {
        hipSetDevice(i);
        checkCudaError(hipMemcpy(h_params + i * paramsPerGPU, d_params[i], paramsPerGPU * sizeof(float), hipMemcpyDeviceToHost), "Failed to copy params to host");
        checkCudaError(hipMemcpy(h_gradients + i * paramsPerGPU, d_gradients[i], paramsPerGPU * sizeof(float), hipMemcpyDeviceToHost), "Failed to copy gradients to host");

        // Free device memory
        checkCudaError(hipFree(d_params[i]), "Failed to free params");
        checkCudaError(hipFree(d_gradients[i]), "Failed to free gradients");
        checkCudaError(hipFree(d_lossGradient[i]), "Failed to free lossGradient");
        checkCudaError(hipFree(d_states[i]), "Failed to free states");
    }

    // Clean up host memory
    delete[] h_params;
    delete[] h_gradients;
    delete[] h_lossGradient;

    std::cout << "Gradient descent completed." << std::endl;
    return 0;
}