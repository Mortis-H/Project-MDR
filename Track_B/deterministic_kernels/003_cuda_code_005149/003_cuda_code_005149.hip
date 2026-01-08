#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define NUM_PARAMETERS 1024
#define BLOCK_SIZE 256

// Kernel to initialize random weights
__global__ void initialize_weights(float *weights, hiprandState *states, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_PARAMETERS) {
        hiprand_init(seed, idx, 0, &states[idx]);
        weights[idx] = hiprand_uniform(&states[idx]) * 2.0f - 1.0f; // Random value between -1 and 1
    }
}

// Kernel to compute the gradient for each parameter
__global__ void compute_gradient(float *weights, float *gradients, float *data, float *labels, int num_samples) {
    extern __shared__ float shared_data[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float grad = 0.0f;

    // Load data into shared memory
    if (idx < num_samples) {
        shared_data[threadIdx.x] = data[idx];
        shared_data[threadIdx.x + blockDim.x] = labels[idx];
        __syncthreads();

        // Compute gradient for each parameter
        for (int i = 0; i < NUM_PARAMETERS; ++i) {
            float prediction = weights[i] * shared_data[threadIdx.x];
            float error = prediction - shared_data[threadIdx.x + blockDim.x];
            grad += error * shared_data[threadIdx.x];
        }
    }

    // Accumulate gradient using atomicAdd
    if (idx < NUM_PARAMETERS) {
        atomicAdd(&gradients[idx], grad);
    }
}

// Kernel to update weights using the computed gradients
__global__ void update_weights(float *weights, float *gradients, float learning_rate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_PARAMETERS) {
        weights[idx] -= learning_rate * gradients[idx];
        gradients[idx] = 0.0f; // Reset gradient for next iteration
    }
}

int main() {
    float *h_weights, *h_gradients, *h_data, *h_labels;
    float *d_weights, *d_gradients, *d_data, *d_labels;
    hiprandState *d_states;

    int num_samples = 1024;
    size_t size_weights = NUM_PARAMETERS * sizeof(float);
    size_t size_gradients = NUM_PARAMETERS * sizeof(float);
    size_t size_data = num_samples * sizeof(float);
    size_t size_labels = num_samples * sizeof(float);
    size_t size_states = NUM_PARAMETERS * sizeof(hiprandState);

    // Allocate host memory
    h_weights = (float *)malloc(size_weights);
    h_gradients = (float *)malloc(size_gradients);
    h_data = (float *)malloc(size_data);
    h_labels = (float *)malloc(size_labels);

    // Allocate device memory
    hipMalloc((void **)&d_weights, size_weights);
    hipMalloc((void **)&d_gradients, size_gradients);
    hipMalloc((void **)&d_data, size_data);
    hipMalloc((void **)&d_labels, size_labels);
    hipMalloc((void **)&d_states, size_states);

    // Initialize device memory
    hipMemset(d_gradients, 0, size_gradients);

    // Initialize weights
    unsigned int seed = 1234;
    initialize_weights<<<(NUM_PARAMETERS + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_weights, d_states, seed);
    hipDeviceSynchronize();

    // Simulate data and labels
    for (int i = 0; i < num_samples; ++i) {
        h_data[i] = static_cast<float>(i) / num_samples;
        h_labels[i] = h_data[i] * 2.0f + 1.0f; // Linear function y = 2x + 1
    }

    // Copy data and labels to device
    hipMemcpy(d_data, h_data, size_data, hipMemcpyHostToDevice);
    hipMemcpy(d_labels, h_labels, size_labels, hipMemcpyHostToDevice);

    // Gradient descent parameters
    float learning_rate = 0.01f;
    int num_iterations = 1000;

    // Perform gradient descent
    for (int iter = 0; iter < num_iterations; ++iter) {
        // Compute gradients
        compute_gradient<<<(num_samples + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 2 * BLOCK_SIZE * sizeof(float)>>>(
            d_weights, d_gradients, d_data, d_labels, num_samples);
        hipDeviceSynchronize();

        // Update weights
        update_weights<<<(NUM_PARAMETERS + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_weights, d_gradients, learning_rate);
        hipDeviceSynchronize();
    }

    // Copy weights back to host
    hipMemcpy(h_weights, d_weights, size_weights, hipMemcpyDeviceToHost);

    // Output final weights
    std::cout << "Final weights:" << std::endl;
    for (int i = 0; i < NUM_PARAMETERS; ++i) {
        std::cout << h_weights[i] << " ";
    }
    std::cout << std::endl;

    // Free memory
    free(h_weights);
    free(h_gradients);
    free(h_data);
    free(h_labels);
    hipFree(d_weights);
    hipFree(d_gradients);
    hipFree(d_data);
    hipFree(d_labels);
    hipFree(d_states);

    return 0;
}