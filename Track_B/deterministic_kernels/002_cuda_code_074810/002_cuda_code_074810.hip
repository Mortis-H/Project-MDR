#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define NUM_VARIABLES 32
#define BLOCK_SIZE 32

// Kernel to perform RMSprop update
__global__ void rmspropUpdate(float* params, float* grads, float* cache, float lr, float decay, float epsilon) {
    extern __shared__ float shared_cache[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < NUM_VARIABLES) {
        // Load cache into shared memory for faster access
        shared_cache[threadIdx.x] = cache[idx];
        __syncthreads();

        // Update cache
        shared_cache[threadIdx.x] = decay * shared_cache[threadIdx.x] + (1.0f - decay) * grads[idx] * grads[idx];
        __syncthreads();

        // Atomic update of cache to global memory
        atomicAdd(&cache[idx], shared_cache[threadIdx.x] - cache[idx]);

        // Compute update step
        float step = -lr * grads[idx] / (sqrtf(cache[idx]) + epsilon);

        // Update parameters
        atomicAdd(&params[idx], step);
    }
}

// Kernel to initialize random gradients and parameters
__global__ void initialize(hiprandState* states, float* params, float* grads, float* cache) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_VARIABLES) {
        hiprandState localState = states[idx];
        float randomValue = hiprand_normal(&localState);
        grads[idx] = randomValue;
        params[idx] = 0.0f;
        cache[idx] = 0.0f;
        states[idx] = localState;
    }
}

int main() {
    // Allocate memory on host
    float* h_params = new float[NUM_VARIABLES];
    float* h_grads = new float[NUM_VARIABLES];
    float* h_cache = new float[NUM_VARIABLES];

    // Allocate memory on device
    float* d_params, *d_grads, *d_cache;
    hiprandState* d_states;
    hipMalloc((void**)&d_params, NUM_VARIABLES * sizeof(float));
    hipMalloc((void**)&d_grads, NUM_VARIABLES * sizeof(float));
    hipMalloc((void**)&d_cache, NUM_VARIABLES * sizeof(float));
    hipMalloc((void**)&d_states, NUM_VARIABLES * sizeof(hiprandState));

    // Initialize random number generator states
    hiprandState* d_states_temp;
    hipMalloc((void**)&d_states_temp, NUM_VARIABLES * sizeof(hiprandState));
    initialize<<<(NUM_VARIABLES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_states_temp, d_params, d_grads, d_cache);
    hipMemcpy(d_states, d_states_temp, NUM_VARIABLES * sizeof(hiprandState), hipMemcpyDeviceToDevice);
    hipFree(d_states_temp);

    // RMSprop parameters
    float learning_rate = 0.001f;
    float decay = 0.9f;
    float epsilon = 1e-8f;

    // Perform RMSprop update
    rmspropUpdate<<<(NUM_VARIABLES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_params, d_grads, d_cache, learning_rate, decay, epsilon);

    // Copy results back to host
    hipMemcpy(h_params, d_params, NUM_VARIABLES * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(h_grads, d_grads, NUM_VARIABLES * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(h_cache, d_cache, NUM_VARIABLES * sizeof(float), hipMemcpyDeviceToHost);

    // Check for errors
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Free device memory
    hipFree(d_params);
    hipFree(d_grads);
    hipFree(d_cache);
    hipFree(d_states);

    // Free host memory
    delete[] h_params;
    delete[] h_grads;
    delete[] h_cache;

    return 0;
}