#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

// Constants for the problem
#define NUM_VARIABLES 8192
#define BLOCK_SIZE 256
#define LEARNING_RATE 0.001f
#define DECAY_RATE 0.9f
#define EPSILON 1e-8f

// Kernel to initialize random values for weights and gradients
__global__ void initialize(hiprandState *state, float *weights, float *gradients, float *cache) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_VARIABLES) {
        hiprandState localState = state[idx];
        weights[idx] = hiprand_normal(&localState) * 0.01f;
        gradients[idx] = 0.0f;
        cache[idx] = 0.0f;
        state[idx] = localState;
    }
}

// Kernel to update weights using RMSprop
__global__ void rmsprop_update(float *weights, float *gradients, float *cache, float learning_rate, float decay_rate, float epsilon) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_VARIABLES) {
        // Update cache
        cache[idx] = decay_rate * cache[idx] + (1.0f - decay_rate) * gradients[idx] * gradients[idx];
        // Update weights
        atomicAdd(&weights[idx], -learning_rate * gradients[idx] / (sqrtf(cache[idx]) + epsilon));
    }
}

// Function to check CUDA errors
void checkCudaError(hipError_t err, const char *file, int line) {
    if (err != hipSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << " - " << hipGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

#define CHECK_CUDA_ERROR(err) checkCudaError(err, __FILE__, __LINE__)

int main() {
    // Allocate memory for weights, gradients, and cache on the host
    float *h_weights = new float[NUM_VARIABLES];
    float *h_gradients = new float[NUM_VARIABLES];
    float *h_cache = new float[NUM_VARIABLES];

    // Allocate memory for weights, gradients, and cache on the device
    float *d_weights, *d_gradients, *d_cache;
    CHECK_CUDA_ERROR(hipMalloc(&d_weights, NUM_VARIABLES * sizeof(float)));
    CHECK_CUDA_ERROR(hipMalloc(&d_gradients, NUM_VARIABLES * sizeof(float)));
    CHECK_CUDA_ERROR(hipMalloc(&d_cache, NUM_VARIABLES * sizeof(float)));

    // Allocate and initialize CURAND states on the device
    hiprandState *d_state;
    CHECK_CUDA_ERROR(hipMalloc(&d_state, NUM_VARIABLES * sizeof(hiprandState)));

    // Initialize CURAND states
    int numBlocks = (NUM_VARIABLES + BLOCK_SIZE - 1) / BLOCK_SIZE;
    initialize<<<numBlocks, BLOCK_SIZE>>>(d_state, d_weights, d_gradients, d_cache);
    CHECK_CUDA_ERROR(hipGetLastError());
    CHECK_CUDA_ERROR(hipDeviceSynchronize());

    // Simulate gradient updates
    for (int i = 0; i < 100; ++i) {
        // Simulate gradient computation (random values for demonstration)
        initialize<<<numBlocks, BLOCK_SIZE>>>(d_state, d_gradients, d_gradients, d_cache);
        CHECK_CUDA_ERROR(hipGetLastError());
        CHECK_CUDA_ERROR(hipDeviceSynchronize());

        // Update weights using RMSprop
        rmsprop_update<<<numBlocks, BLOCK_SIZE>>>(d_weights, d_gradients, d_cache, LEARNING_RATE, DECAY_RATE, EPSILON);
        CHECK_CUDA_ERROR(hipGetLastError());
        CHECK_CUDA_ERROR(hipDeviceSynchronize());
    }

    // Copy results back to host
    CHECK_CUDA_ERROR(hipMemcpy(h_weights, d_weights, NUM_VARIABLES * sizeof(float), hipMemcpyDeviceToHost));

    // Clean up
    delete[] h_weights;
    delete[] h_gradients;
    delete[] h_cache;
    CHECK_CUDA_ERROR(hipFree(d_weights));
    CHECK_CUDA_ERROR(hipFree(d_gradients));
    CHECK_CUDA_ERROR(hipFree(d_cache));
    CHECK_CUDA_ERROR(hipFree(d_state));

    std::cout << "RMSprop update completed successfully." << std::endl;
    return 0;
}