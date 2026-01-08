#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define BLOCK_SIZE 256
#define NUM_PARAMS 256
#define LEARNING_RATE 0.01f
#define ITERATIONS 1000

__global__ void initializeParameters(hiprandState *state, float *params) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_PARAMS) {
        hiprand_init(1234, idx, 0, &state[idx]);
        params[idx] = hiprand_uniform(&state[idx]) * 10.0f; // Random initialization
    }
}

__global__ void gradientDescentKernel(float *params, float *gradients, float learningRate) {
    extern __shared__ float sharedParams[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load parameters into shared memory
    if (idx < NUM_PARAMS) {
        sharedParams[threadIdx.x] = params[idx];
    }
    __syncthreads();

    // Compute gradient (dummy example: gradient is the negative of the parameter)
    float gradient = -sharedParams[threadIdx.x];

    // Update parameters
    if (idx < NUM_PARAMS) {
        params[idx] -= learningRate * gradient;
    }
}

void checkCudaError(hipError_t err, const char *msg) {
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << msg << " (" << hipGetErrorString(err) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    int numDevices;
    hipGetDeviceCount(&numDevices);
    if (numDevices < 1) {
        std::cerr << "No CUDA devices found." << std::endl;
        return EXIT_FAILURE;
    }

    // Allocate memory for parameters and states on the host
    float *h_params = new float[NUM_PARAMS];
    hiprandState *h_states = new hiprandState[NUM_PARAMS];

    // Allocate memory for parameters and states on the device
    float *d_params;
    hiprandState *d_states;
    checkCudaError(hipMalloc((void**)&d_params, NUM_PARAMS * sizeof(float)), "hipMalloc d_params");
    checkCudaError(hipMalloc((void**)&d_states, NUM_PARAMS * sizeof(hiprandState)), "hipMalloc d_states");

    // Initialize parameters on the device
    initializeParameters<<<(NUM_PARAMS + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_states, d_params);
    checkCudaError(hipGetLastError(), "initializeParameters kernel launch");
    checkCudaError(hipDeviceSynchronize(), "initializeParameters sync");

    // Allocate memory for gradients on the device
    float *d_gradients;
    checkCudaError(hipMalloc((void**)&d_gradients, NUM_PARAMS * sizeof(float)), "hipMalloc d_gradients");

    // Perform gradient descent
    for (int iter = 0; iter < ITERATIONS; ++iter) {
        gradientDescentKernel<<<(NUM_PARAMS + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_params, d_gradients, LEARNING_RATE);
        checkCudaError(hipGetLastError(), "gradientDescentKernel launch");
        checkCudaError(hipDeviceSynchronize(), "gradientDescentKernel sync");
    }

    // Copy parameters back to host
    checkCudaError(hipMemcpy(h_params, d_params, NUM_PARAMS * sizeof(float), hipMemcpyDeviceToHost), "hipMemcpy h_params");

    // Free device memory
    checkCudaError(hipFree(d_params), "hipFree d_params");
    checkCudaError(hipFree(d_states), "hipFree d_states");
    checkCudaError(hipFree(d_gradients), "hipFree d_gradients");

    // Free host memory
    delete[] h_params;
    delete[] h_states;

    std::cout << "Gradient descent completed successfully." << std::endl;
    return EXIT_SUCCESS;
}