#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define LAYERS 5
#define NEURONS 256
#define BLOCK_SIZE 128

// Kernel to initialize weights with random values
__global__ void initializeWeights(float *weights, int size) {
    hiprandState state;
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    hiprand_init(id, 0, 0, &state);
    if (id < size) {
        weights[id] = hiprand_uniform(&state) * 0.1f;
    }
}

// Kernel to perform forward pass
__global__ void forwardPass(float *input, float *weights, float *output, int layer) {
    extern __shared__ float sharedData[];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int neuronIdx = bid * blockDim.x + tid;

    // Load input data into shared memory
    if (neuronIdx < NEURONS) {
        sharedData[tid] = input[neuronIdx];
    }
    __syncthreads();

    // Compute weighted sum for each neuron
    float sum = 0.0f;
    for (int i = 0; i < NEURONS; i++) {
        sum += sharedData[i] * weights[neuronIdx * NEURONS + i];
    }
    __syncthreads();

    // Apply activation function (ReLU)
    if (neuronIdx < NEURONS) {
        output[neuronIdx] = sum > 0.0f ? sum : 0.0f;
    }
}

// Kernel to perform backward pass
__global__ void backwardPass(float *output, float *weights, float *gradOutput, float *gradWeights, int layer) {
    extern __shared__ float sharedData[];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int neuronIdx = bid * blockDim.x + tid;

    // Load gradient data into shared memory
    if (neuronIdx < NEURONS) {
        sharedData[tid] = gradOutput[neuronIdx];
    }
    __syncthreads();

    // Compute gradient for weights
    float gradSum = 0.0f;
    for (int i = 0; i < NEURONS; i++) {
        gradSum += sharedData[i] * weights[neuronIdx * NEURONS + i];
    }
    __syncthreads();

    // Update gradient weights
    if (neuronIdx < NEURONS) {
        gradWeights[neuronIdx] = gradSum;
    }
}

// Main function to demonstrate the usage of the kernels
int main() {
    float *d_input, *d_weights[LAYERS], *d_output[LAYERS], *d_gradOutput[LAYERS], *d_gradWeights[LAYERS];
    float *h_input = new float[NEURONS];
    float *h_weights[LAYERS], *h_output[LAYERS], *h_gradOutput[LAYERS], *h_gradWeights[LAYERS];

    // Initialize input data
    for (int i = 0; i < NEURONS; i++) {
        h_input[i] = 1.0f; // Example input
    }

    // Allocate memory on host and device
    for (int i = 0; i < LAYERS; i++) {
        h_weights[i] = new float[NEURONS * NEURONS];
        h_output[i] = new float[NEURONS];
        h_gradOutput[i] = new float[NEURONS];
        h_gradWeights[i] = new float[NEURONS * NEURONS];

        hipMalloc(&d_weights[i], NEURONS * NEURONS * sizeof(float));
        hipMalloc(&d_output[i], NEURONS * sizeof(float));
        hipMalloc(&d_gradOutput[i], NEURONS * sizeof(float));
        hipMalloc(&d_gradWeights[i], NEURONS * NEURONS * sizeof(float));
    }

    // Initialize weights with random values
    initializeWeights<<<(NEURONS * NEURONS + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_weights[0], NEURONS * NEURONS);
    hipDeviceSynchronize();

    // Copy input data to device
    hipMemcpy(d_input, h_input, NEURONS * sizeof(float), hipMemcpyHostToDevice);

    // Perform forward pass
    for (int i = 0; i < LAYERS; i++) {
        if (i == 0) {
            forwardPass<<<(NEURONS + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_input, d_weights[i], d_output[i], i);
        } else {
            forwardPass<<<(NEURONS + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_output[i - 1], d_weights[i], d_output[i], i);
        }
        hipDeviceSynchronize();
    }

    // Initialize gradient output (example)
    for (int i = 0; i < NEURONS; i++) {
        h_gradOutput[LAYERS - 1][i] = 1.0f; // Example gradient
    }
    hipMemcpy(d_gradOutput[LAYERS - 1], h_gradOutput[LAYERS - 1], NEURONS * sizeof(float), hipMemcpyHostToDevice);

    // Perform backward pass
    for (int i = LAYERS - 1; i >= 0; i--) {
        if (i == LAYERS - 1) {
            backwardPass<<<(NEURONS + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_output[i], d_weights[i], d_gradOutput[i], d_gradWeights[i], i);
        } else {
            backwardPass<<<(NEURONS + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_output[i], d_weights[i], d_gradOutput[i + 1], d_gradWeights[i], i);
        }
        hipDeviceSynchronize();
    }

    // Free allocated memory
    for (int i = 0; i < LAYERS; i++) {
        delete[] h_weights[i];
        delete[] h_output[i];
        delete[] h_gradOutput[i];
        delete[] h_gradWeights[i];

        hipFree(d_weights[i]);
        hipFree(d_output[i]);
        hipFree(d_gradOutput[i]);
        hipFree(d_gradWeights[i]);
    }
    delete[] h_input;

    // Check for any errors launching the kernel
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    std::cout << "CUDA program executed successfully!" << std::endl;
    return 0;
}