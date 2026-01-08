#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

// Constants for Adam optimizer
#define NUM_VARIABLES 2048
#define BLOCK_SIZE 256
#define NUM_BLOCKS (NUM_VARIABLES + BLOCK_SIZE - 1) / BLOCK_SIZE
#define BETA1 0.9f
#define BETA2 0.999f
#define EPSILON 1e-8f
#define LEARNING_RATE 0.001f

// Kernel to perform Adam update
__global__ void adamUpdateKernel(float* variables, float* gradients, float* m, float* v, float t) {
    extern __shared__ float shared_grads[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_VARIABLES) return;

    // Load gradient into shared memory for coalesced access
    shared_grads[threadIdx.x] = gradients[idx];
    __syncthreads();

    // Compute first moment (running average of the gradients)
    m[idx] = BETA1 * m[idx] + (1.0f - BETA1) * shared_grads[threadIdx.x];

    // Compute second moment (running variance of the gradients)
    v[idx] = BETA2 * v[idx] + (1.0f - BETA2) * shared_grads[threadIdx.x] * shared_grads[threadIdx.x];

    // Compute bias-corrected first moment estimate
    float m_hat = m[idx] / (1.0f - pow(BETA1, t));

    // Compute bias-corrected second raw moment estimate
    float v_hat = v[idx] / (1.0f - pow(BETA2, t));

    // Update the variables
    variables[idx] -= LEARNING_RATE * m_hat / (sqrt(v_hat) + EPSILON);

    __syncthreads();
}

// Function to initialize random data
__global__ void initializeData(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = static_cast<float>(idx) / static_cast<float>(size);
    }
}

// Error checking macro
#define CUDA_CHECK(call)                                                            \
    {                                                                               \
        hipError_t err = call;                                                     \
        if (err != hipSuccess) {                                                   \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - "     \
                      << hipGetErrorString(err) << std::endl;                        \
            exit(EXIT_FAILURE);                                                     \
        }                                                                           \
    }

int main() {
    // Allocate memory on host
    float* h_variables = new float[NUM_VARIABLES];
    float* h_gradients = new float[NUM_VARIABLES];
    float* h_m = new float[NUM_VARIABLES];
    float* h_v = new float[NUM_VARIABLES];

    // Allocate memory on device
    float* d_variables, *d_gradients, *d_m, *d_v;
    CUDA_CHECK(hipMalloc(&d_variables, NUM_VARIABLES * sizeof(float)));
    CUDA_CHECK(hipMalloc(&d_gradients, NUM_VARIABLES * sizeof(float)));
    CUDA_CHECK(hipMalloc(&d_m, NUM_VARIABLES * sizeof(float)));
    CUDA_CHECK(hipMalloc(&d_v, NUM_VARIABLES * sizeof(float)));

    // Initialize data on device
    initializeData<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_variables, NUM_VARIABLES);
    initializeData<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_gradients, NUM_VARIABLES);
    initializeData<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_m, NUM_VARIABLES);
    initializeData<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_v, NUM_VARIABLES);
    CUDA_CHECK(hipDeviceSynchronize());

    // Perform Adam update
    float t = 1.0f; // Time step
    adamUpdateKernel<<<NUM_BLOCKS, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_variables, d_gradients, d_m, d_v, t);
    CUDA_CHECK(hipDeviceSynchronize());

    // Copy results back to host
    CUDA_CHECK(hipMemcpy(h_variables, d_variables, NUM_VARIABLES * sizeof(float), hipMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(hipFree(d_variables));
    CUDA_CHECK(hipFree(d_gradients));
    CUDA_CHECK(hipFree(d_m));
    CUDA_CHECK(hipFree(d_v));

    // Free host memory
    delete[] h_variables;
    delete[] h_gradients;
    delete[] h_m;
    delete[] h_v;

    return 0;
}