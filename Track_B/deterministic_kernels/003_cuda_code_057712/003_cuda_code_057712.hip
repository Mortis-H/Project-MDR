#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define NUM_LAYERS 5
#define NUM_NEURONS 256
#define BATCH_SIZE 256

// Kernel to perform matrix multiplication for neural network layers
__global__ void matrixMultiply(float* A, float* B, float* C, int numAcols, int numBcols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    if (row < NUM_NEURONS && col < numBcols) {
        for (int k = 0; k < numAcols; ++k) {
            sum += A[row * numAcols + k] * B[k * numBcols + col];
        }
        C[row * numBcols + col] = sum;
    }
}

// Kernel to apply ReLU activation function
__global__ void applyReLU(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

// Kernel to initialize weights with random values
__global__ void initializeWeights(float* weights, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    hiprandState state;
    hiprand_init((unsigned long long) clock() + idx, 0, 0, &state);
    if (idx < size) {
        weights[idx] = hiprand_uniform(&state) * 0.01f;
    }
}

int main() {
    // Allocate memory for weights and data
    float* h_weights[NUM_LAYERS];
    float* h_data;
    float* d_weights[NUM_LAYERS];
    float* d_data;

    // Allocate host memory
    for (int i = 0; i < NUM_LAYERS; ++i) {
        h_weights[i] = (float*)malloc(NUM_NEURONS * NUM_NEURONS * sizeof(float));
    }
    h_data = (float*)malloc(NUM_NEURONS * BATCH_SIZE * sizeof(float));

    // Initialize weights on host
    for (int i = 0; i < NUM_LAYERS; ++i) {
        for (int j = 0; j < NUM_NEURONS * NUM_NEURONS; ++j) {
            h_weights[i][j] = 0.01f;
        }
    }

    // Allocate device memory
    for (int i = 0; i < NUM_LAYERS; ++i) {
        hipMalloc(&d_weights[i], NUM_NEURONS * NUM_NEURONS * sizeof(float));
        hipMemcpy(d_weights[i], h_weights[i], NUM_NEURONS * NUM_NEURONS * sizeof(float), hipMemcpyHostToDevice);
    }
    hipMalloc(&d_data, NUM_NEURONS * BATCH_SIZE * sizeof(float));

    // Initialize input data on device
    initializeWeights<<<(NUM_NEURONS * BATCH_SIZE + 255) / 256, 256>>>(d_data, NUM_NEURONS * BATCH_SIZE);
    hipDeviceSynchronize();

    // Define block and grid sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((NUM_NEURONS + blockSize.x - 1) / blockSize.x, (NUM_NEURONS + blockSize.y - 1) / blockSize.y);

    // Forward pass through the network
    for (int i = 0; i < NUM_LAYERS; ++i) {
        float* input = (i == 0) ? d_data : d_weights[i - 1];
        matrixMultiply<<<gridSize, blockSize>>>(input, d_weights[i], d_weights[i], NUM_NEURONS, NUM_NEURONS);
        hipDeviceSynchronize();
        applyReLU<<<(NUM_NEURONS * NUM_NEURONS + 255) / 256, 256>>>(d_weights[i], NUM_NEURONS * NUM_NEURONS);
        hipDeviceSynchronize();
    }

    // Free device memory
    for (int i = 0; i < NUM_LAYERS; ++i) {
        hipFree(d_weights[i]);
    }
    hipFree(d_data);

    // Free host memory
    for (int i = 0; i < NUM_LAYERS; ++i) {
        free(h_weights[i]);
    }
    free(h_data);

    // Check for any errors launching the kernel
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch kernel (error code " << err << " : " << hipGetErrorString(err) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "Neural network forward pass completed successfully." << std::endl;

    return 0;
}