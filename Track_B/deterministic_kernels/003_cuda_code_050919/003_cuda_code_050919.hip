#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define N 64
#define BATCH_SIZE 256
#define LEARNING_RATE 0.01f

// Kernel to initialize weights and biases
__global__ void initialize_weights(float *weights, float *biases, int size) {
    hiprandState state;
    hiprand_init((unsigned long)clock() + threadIdx.x, blockIdx.x, 0, &state);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] = hiprand_uniform(&state) * 0.01f;
        biases[idx] = hiprand_uniform(&state) * 0.01f;
    }
}

// Sigmoid activation function
__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Derivative of sigmoid function
__device__ float sigmoid_derivative(float x) {
    float s = sigmoid(x);
    return s * (1.0f - s);
}

// Forward pass kernel
__global__ void forward_pass(float *input, float *weights1, float *biases1, float *weights2, float *biases2, float *output, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        float z1 = 0.0f, z2 = 0.0f;
        for (int i = 0; i < N; ++i) {
            z1 += input[idx * N + i] * weights1[i];
        }
        z1 += biases1[0];
        float a1 = sigmoid(z1);

        for (int i = 0; i < N; ++i) {
            z2 += a1 * weights2[i];
        }
        z2 += biases2[0];
        output[idx] = sigmoid(z2);
    }
}

// Backward pass kernel
__global__ void backward_pass(float *input, float *weights1, float *biases1, float *weights2, float *biases2, float *output, float *target, int batch_size, float learning_rate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        float z1 = 0.0f, z2 = 0.0f;
        for (int i = 0; i < N; ++i) {
            z1 += input[idx * N + i] * weights1[i];
        }
        z1 += biases1[0];
        float a1 = sigmoid(z1);

        for (int i = 0; i < N; ++i) {
            z2 += a1 * weights2[i];
        }
        z2 += biases2[0];
        float a2 = sigmoid(z2);

        float error = a2 - target[idx];
        float delta2 = error * sigmoid_derivative(a2);

        float delta1[N];
        for (int i = 0; i < N; ++i) {
            delta1[i] = delta2 * weights2[i] * sigmoid_derivative(z1);
        }

        // Update weights and biases
        for (int i = 0; i < N; ++i) {
            weights2[i] -= learning_rate * a1 * delta2;
        }
        biases2[0] -= learning_rate * delta2;

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                weights1[i] -= learning_rate * input[idx * N + j] * delta1[i];
            }
        }
        biases1[0] -= learning_rate * delta1[0];
    }
}

int main() {
    float *h_input, *h_target, *h_weights1, *h_weights2, *h_biases1, *h_biases2, *h_output;
    float *d_input, *d_target, *d_weights1, *d_weights2, *d_biases1, *d_biases2, *d_output;

    // Allocate host memory
    h_input = (float *)malloc(BATCH_SIZE * N * sizeof(float));
    h_target = (float *)malloc(BATCH_SIZE * sizeof(float));
    h_weights1 = (float *)malloc(N * sizeof(float));
    h_weights2 = (float *)malloc(N * sizeof(float));
    h_biases1 = (float *)malloc(sizeof(float));
    h_biases2 = (float *)malloc(sizeof(float));
    h_output = (float *)malloc(BATCH_SIZE * sizeof(float));

    // Initialize host data
    for (int i = 0; i < BATCH_SIZE * N; ++i) h_input[i] = (float)(i % 2);
    for (int i = 0; i < BATCH_SIZE; ++i) h_target[i] = (float)(i % 2);

    // Allocate device memory
    hipMalloc((void **)&d_input, BATCH_SIZE * N * sizeof(float));
    hipMalloc((void **)&d_target, BATCH_SIZE * sizeof(float));
    hipMalloc((void **)&d_weights1, N * sizeof(float));
    hipMalloc((void **)&d_weights2, N * sizeof(float));
    hipMalloc((void **)&d_biases1, sizeof(float));
    hipMalloc((void **)&d_biases2, sizeof(float));
    hipMalloc((void **)&d_output, BATCH_SIZE * sizeof(float));

    // Copy data from host to device
    hipMemcpy(d_input, h_input, BATCH_SIZE * N * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_target, h_target, BATCH_SIZE * sizeof(float), hipMemcpyHostToDevice);

    // Initialize weights and biases
    initialize_weights<<<1, N>>>(d_weights1, d_biases1, N);
    initialize_weights<<<1, N>>>(d_weights2, d_biases2, N);
    hipDeviceSynchronize();

    // Set up grid and block dimensions
    dim3 blockSize(256);
    dim3 gridSize((BATCH_SIZE + blockSize.x - 1) / blockSize.x);

    // Perform forward and backward passes
    for (int epoch = 0; epoch < 1000; ++epoch) {
        forward_pass<<<gridSize, blockSize>>>(d_input, d_weights1, d_biases1, d_weights2, d_biases2, d_output, BATCH_SIZE);
        backward_pass<<<gridSize, blockSize>>>(d_input, d_weights1, d_biases1, d_weights2, d_biases2, d_output, d_target, BATCH_SIZE, LEARNING_RATE);
        hipDeviceSynchronize();
    }

    // Copy output back to host
    hipMemcpy(h_output, d_output, BATCH_SIZE * sizeof(float), hipMemcpyDeviceToHost);

    // Free device memory
    hipFree(d_input);
    hipFree(d_target);
    hipFree(d_weights1);
    hipFree(d_weights2);
    hipFree(d_biases1);
    hipFree(d_biases2);
    hipFree(d_output);

    // Free host memory
    free(h_input);
    free(h_target);
    free(h_weights1);
    free(h_weights2);
    free(h_biases1);
    free(h_biases2);
    free(h_output);

    return 0;
}