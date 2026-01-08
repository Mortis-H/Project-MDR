// gradient_descent.cu
#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define NUM_PARAMETERS 32
#define NUM_GPUS 2
#define BLOCK_SIZE 256

// Kernel to perform gradient descent update
__global__ void gradientDescentKernel(float* parameters, const float* gradients, float learningRate, int numParams) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numParams) {
        // Avoid bank conflicts by using shared memory
        __shared__ float s_params[BLOCK_SIZE];
        s_params[threadIdx.x] = parameters[idx];
        __syncthreads();

        // Update parameters
        s_params[threadIdx.x] -= learningRate * gradients[idx];
        __syncthreads();

        parameters[idx] = s_params[threadIdx.x];
    }
}

// Function to initialize random gradients for demonstration
__global__ void initializeGradients(float* gradients, int numParams) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numParams) {
        hiprandState state;
        hiprand_init((unsigned long long) clock() + idx, 0, 0, &state);
        gradients[idx] = hiprand_uniform(&state) * 2.0f - 1.0f; // Random value between -1 and 1
    }
}

// Error checking macro
#define CUDA_CHECK(call)                                                            \
    do {                                                                            \
        hipError_t err = call;                                                     \
        if (err != hipSuccess) {                                                   \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - "    \
                      << hipGetErrorString(err) << std::endl;                      \
            exit(EXIT_FAILURE);                                                     \
        }                                                                           \
    } while (0)

int main() {
    float* h_parameters = new float[NUM_PARAMETERS];
    float* h_gradients = new float[NUM_PARAMETERS];
    float learningRate = 0.01f;

    // Initialize parameters to zero
    for (int i = 0; i < NUM_PARAMETERS; ++i) {
        h_parameters[i] = 0.0f;
    }

    // Allocate device memory
    float* d_parameters[NUM_GPUS];
    float* d_gradients[NUM_GPUS];
    for (int gpu = 0; gpu < NUM_GPUS; ++gpu) {
        CUDA_CHECK(hipSetDevice(gpu));
        CUDA_CHECK(hipMalloc(&d_parameters[gpu], NUM_PARAMETERS * sizeof(float)));
        CUDA_CHECK(hipMalloc(&d_gradients[gpu], NUM_PARAMETERS * sizeof(float)));
        CUDA_CHECK(hipMemcpy(d_parameters[gpu], h_parameters, NUM_PARAMETERS * sizeof(float), hipMemcpyHostToDevice));
    }

    // Initialize gradients
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (NUM_PARAMETERS + threadsPerBlock - 1) / threadsPerBlock;
    for (int gpu = 0; gpu < NUM_GPUS; ++gpu) {
        CUDA_CHECK(hipSetDevice(gpu));
        initializeGradients<<<blocksPerGrid, threadsPerBlock>>>(d_gradients[gpu], NUM_PARAMETERS);
        CUDA_CHECK(hipDeviceSynchronize());
    }

    // Perform gradient descent
    for (int iter = 0; iter < 100; ++iter) {
        for (int gpu = 0; gpu < NUM_GPUS; ++gpu) {
            CUDA_CHECK(hipSetDevice(gpu));
            gradientDescentKernel<<<blocksPerGrid, threadsPerBlock>>>(d_parameters[gpu], d_gradients[gpu], learningRate, NUM_PARAMETERS);
            CUDA_CHECK(hipDeviceSynchronize());
        }
    }

    // Copy results back to host
    for (int gpu = 0; gpu < NUM_GPUS; ++gpu) {
        CUDA_CHECK(hipSetDevice(gpu));
        CUDA_CHECK(hipMemcpy(h_parameters, d_parameters[gpu], NUM_PARAMETERS * sizeof(float), hipMemcpyDeviceToHost));
    }

    // Free device memory
    for (int gpu = 0; gpu < NUM_GPUS; ++gpu) {
        CUDA_CHECK(hipSetDevice(gpu));
        CUDA_CHECK(hipFree(d_parameters[gpu]));
        CUDA_CHECK(hipFree(d_gradients[gpu]));
    }

    // Free host memory
    delete[] h_parameters;
    delete[] h_gradients;

    return 0;
}