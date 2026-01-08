#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

// Define the number of parameters
#define NUM_PARAMETERS 32

// Kernel to initialize random parameters
__global__ void initParameters(float *params, hiprandState *states, int seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < NUM_PARAMETERS) {
        hiprandState localState = states[idx];
        hiprand_init(seed, idx, 0, &localState);
        params[idx] = hiprand_uniform(&localState) * 10.0f; // Initialize parameters between 0 and 10
        states[idx] = localState;
    }
}

// Kernel to compute the gradient (dummy function for demonstration)
__global__ void computeGradient(float *params, float *grad) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < NUM_PARAMETERS) {
        grad[idx] = 2.0f * params[idx]; // Example gradient: f(x) = x^2, df/dx = 2x
    }
}

// Kernel to update parameters using gradient descent
__global__ void updateParameters(float *params, float *grad, float learningRate) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < NUM_PARAMETERS) {
        params[idx] -= learningRate * grad[idx];
    }
}

int main() {
    const int numThreads = 256;
    const int numBlocks = (NUM_PARAMETERS + numThreads - 1) / numThreads;
    const float learningRate = 0.01f;

    float *d_params, *d_grad;
    hiprandState *d_states;

    // Allocate memory on the device
    hipError_t err = hipMalloc((void**)&d_params, NUM_PARAMETERS * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate memory for parameters: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc((void**)&d_grad, NUM_PARAMETERS * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate memory for gradients: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc((void**)&d_states, NUM_PARAMETERS * sizeof(hiprandState));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate memory for states: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Initialize parameters
    initParameters<<<numBlocks, numThreads>>>(d_params, d_states, 1234);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to initialize parameters: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Perform gradient descent iterations
    for (int i = 0; i < 100; ++i) {
        // Compute gradient
        computeGradient<<<numBlocks, numThreads>>>(d_params, d_grad);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Failed to compute gradient: " << hipGetErrorString(err) << std::endl;
            return -1;
        }

        // Update parameters
        updateParameters<<<numBlocks, numThreads>>>(d_params, d_grad, learningRate);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Failed to update parameters: " << hipGetErrorString(err) << std::endl;
            return -1;
        }
    }

    // Copy parameters back to host for verification
    float h_params[NUM_PARAMETERS];
    err = hipMemcpy(h_params, d_params, NUM_PARAMETERS * sizeof(float), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy parameters to host: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Print final parameters
    std::cout << "Final parameters:" << std::endl;
    for (int i = 0; i < NUM_PARAMETERS; ++i) {
        std::cout << h_params[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    hipFree(d_params);
    hipFree(d_grad);
    hipFree(d_states);

    return 0;
}