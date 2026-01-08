#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

// Define the number of layers and neurons per layer
#define NUM_LAYERS 4
#define NEURONS_PER_LAYER 128

// Define the batch size for training
#define BATCH_SIZE 256

// Kernel to perform matrix multiplication for neural network layers
__global__ void matrixMultiply(float* A, float* B, float* C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    if (row < width && col < width) {
        for (int k = 0; k < width; ++k) {
            sum += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = sum;
    }
}

// Kernel to add bias and apply ReLU activation
__global__ void addBiasAndReLU(float* A, float* bias, float* C, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width) {
        C[idx] = fmaxf(0.0f, A[idx] + bias[idx]);
    }
}

// Initialize weights and biases using random numbers
__global__ void initWeightsAndBiases(float* weights, float* biases, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        hiprandState state;
        hiprand_init((unsigned long long)clock() + idx, 0, 0, &state);
        weights[idx] = hiprand_uniform(&state) * 0.1f;
        biases[idx] = hiprand_uniform(&state) * 0.1f;
    }
}

int main() {
    // Allocate host memory
    float* h_input = new float[BATCH_SIZE * NEURONS_PER_LAYER];
    float* h_output = new float[BATCH_SIZE * NEURONS_PER_LAYER];
    float* h_weights[NUM_LAYERS];
    float* h_biases[NUM_LAYERS];

    for (int i = 0; i < NUM_LAYERS; ++i) {
        h_weights[i] = new float[NEURONS_PER_LAYER * NEURONS_PER_LAYER];
        h_biases[i] = new float[NEURONS_PER_LAYER];
    }

    // Allocate device memory
    float* d_input;
    float* d_output;
    float* d_weights[NUM_LAYERS];
    float* d_biases[NUM_LAYERS];

    hipError_t err = hipMalloc((void**)&d_input, BATCH_SIZE * NEURONS_PER_LAYER * sizeof(float));
    if (err != hipSuccess) { std::cerr << "Error allocating d_input: " << hipGetErrorString(err) << std::endl; return -1; }

    err = hipMalloc((void**)&d_output, BATCH_SIZE * NEURONS_PER_LAYER * sizeof(float));
    if (err != hipSuccess) { std::cerr << "Error allocating d_output: " << hipGetErrorString(err) << std::endl; return -1; }

    for (int i = 0; i < NUM_LAYERS; ++i) {
        err = hipMalloc((void**)&d_weights[i], NEURONS_PER_LAYER * NEURONS_PER_LAYER * sizeof(float));
        if (err != hipSuccess) { std::cerr << "Error allocating d_weights[" << i << "]: " << hipGetErrorString(err) << std::endl; return -1; }

        err = hipMalloc((void**)&d_biases[i], NEURONS_PER_LAYER * sizeof(float));
        if (err != hipSuccess) { std::cerr << "Error allocating d_biases[" << i << "]: " << hipGetErrorString(err) << std::endl; return -1; }
    }

    // Initialize weights and biases on the device
    dim3 blockSize(256);
    dim3 gridSize((NEURONS_PER_LAYER * NEURONS_PER_LAYER + blockSize.x - 1) / blockSize.x);
    initWeightsAndBiases<<<gridSize, blockSize>>>(d_weights[0], d_biases[0], NEURONS_PER_LAYER * NEURONS_PER_LAYER);
    hipDeviceSynchronize();

    // Copy weights and biases to device
    for (int i = 1; i < NUM_LAYERS; ++i) {
        hipMemcpy(d_weights[i], d_weights[0], NEURONS_PER_LAYER * NEURONS_PER_LAYER * sizeof(float), hipMemcpyDeviceToDevice);
        hipMemcpy(d_biases[i], d_biases[0], NEURONS_PER_LAYER * sizeof(float), hipMemcpyDeviceToDevice);
    }

    // Initialize input data on the host
    for (int i = 0; i < BATCH_SIZE * NEURONS_PER_LAYER; ++i) {
        h_input[i] = 0.5f; // Example input data
    }

    // Copy input data to device
    hipMemcpy(d_input, h_input, BATCH_SIZE * NEURONS_PER_LAYER * sizeof(float), hipMemcpyHostToDevice);

    // Define grid and block sizes for matrix multiplication
    dim3 matMulGridSize((NEURONS_PER_LAYER + blockSize.x - 1) / blockSize.x, (BATCH_SIZE + blockSize.y - 1) / blockSize.y);

    // Define grid and block sizes for bias addition and ReLU
    dim3 biasReLUGridSize((NEURONS_PER_LAYER + blockSize.x - 1) / blockSize.x);

    // Perform forward pass
    for (int i = 0; i < NUM_LAYERS; ++i) {
        if (i == 0) {
            matrixMultiply<<<matMulGridSize, blockSize>>>(d_input, d_weights[i], d_output, NEURONS_PER_LAYER);
        } else {
            matrixMultiply<<<matMulGridSize, blockSize>>>(d_output, d_weights[i], d_input, NEURONS_PER_LAYER);
        }
        hipDeviceSynchronize();

        if (i < NUM_LAYERS - 1) {
            addBiasAndReLU<<<biasReLUGridSize, blockSize>>>(d_input, d_biases[i], d_output, NEURONS_PER_LAYER);
            hipDeviceSynchronize();
        }
    }

    // Copy output data to host
    hipMemcpy(h_output, d_output, BATCH_SIZE * NEURONS_PER_LAYER * sizeof(float), hipMemcpyDeviceToHost);

    // Free device memory
    hipFree(d_input);
    hipFree(d_output);
    for (int i = 0; i < NUM_LAYERS; ++i) {
        hipFree(d_weights[i]);
        hipFree(d_biases[i]);
    }

    // Free host memory
    delete[] h_input;
    delete[] h_output;
    for (int i = 0; i < NUM_LAYERS; ++i) {
        delete[] h_weights[i];
        delete[] h_biases[i];
    }

    return 0;
}