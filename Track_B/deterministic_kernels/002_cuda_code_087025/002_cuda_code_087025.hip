#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define NUM_LAYERS 8
#define NUM_NEURONS 256
#define BATCH_SIZE 1024

// Simple sigmoid activation function
__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Kernel to perform forward pass
__global__ void forwardPass(float* input, float* weights, float* biases, float* output) {
    int neuronIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuronIdx >= NUM_NEURONS) return;

    float sum = biases[neuronIdx];
    for (int i = 0; i < NUM_NEURONS; ++i) {
        sum += input[i] * weights[neuronIdx * NUM_NEURONS + i];
    }
    output[neuronIdx] = sigmoid(sum);
}

// Kernel to perform backpropagation
__global__ void backpropagation(float* input, float* weights, float* biases, float* output, float* error, float* weightDeltas, float* biasDeltas, float learningRate) {
    int neuronIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuronIdx >= NUM_NEURONS) return;

    float outputError = error[neuronIdx];
    float delta = outputError * output[neuronIdx] * (1.0f - output[neuronIdx]);

    for (int i = 0; i < NUM_NEURONS; ++i) {
        weightDeltas[neuronIdx * NUM_NEURONS + i] += input[i] * delta;
    }
    biasDeltas[neuronIdx] += delta;

    // Update weights and biases
    for (int i = 0; i < NUM_NEURONS; ++i) {
        weights[neuronIdx * NUM_NEURONS + i] -= learningRate * weightDeltas[neuronIdx * NUM_NEURONS + i];
    }
    biases[neuronIdx] -= learningRate * biasDeltas[neuronIdx];
}

int main() {
    float *h_input, *h_weights, *h_biases, *h_output, *h_error, *h_weightDeltas, *h_biasDeltas;
    float *d_input, *d_weights, *d_biases, *d_output, *d_error, *d_weightDeltas, *d_biasDeltas;

    // Allocate host memory
    h_input = (float*)malloc(BATCH_SIZE * sizeof(float));
    h_weights = (float*)malloc(NUM_LAYERS * NUM_NEURONS * NUM_NEURONS * sizeof(float));
    h_biases = (float*)malloc(NUM_LAYERS * NUM_NEURONS * sizeof(float));
    h_output = (float*)malloc(BATCH_SIZE * sizeof(float));
    h_error = (float*)malloc(BATCH_SIZE * sizeof(float));
    h_weightDeltas = (float*)malloc(NUM_LAYERS * NUM_NEURONS * NUM_NEURONS * sizeof(float));
    h_biasDeltas = (float*)malloc(NUM_LAYERS * NUM_NEURONS * sizeof(float));

    // Initialize host memory
    for (int i = 0; i < BATCH_SIZE; ++i) h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < NUM_LAYERS * NUM_NEURONS * NUM_NEURONS; ++i) h_weights[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < NUM_LAYERS * NUM_NEURONS; ++i) h_biases[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < BATCH_SIZE; ++i) h_error[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < NUM_LAYERS * NUM_NEURONS * NUM_NEURONS; ++i) h_weightDeltas[i] = 0.0f;
    for (int i = 0; i < NUM_LAYERS * NUM_NEURONS; ++i) h_biasDeltas[i] = 0.0f;

    // Allocate device memory
    hipMalloc((void**)&d_input, BATCH_SIZE * sizeof(float));
    hipMalloc((void**)&d_weights, NUM_LAYERS * NUM_NEURONS * NUM_NEURONS * sizeof(float));
    hipMalloc((void**)&d_biases, NUM_LAYERS * NUM_NEURONS * sizeof(float));
    hipMalloc((void**)&d_output, BATCH_SIZE * sizeof(float));
    hipMalloc((void**)&d_error, BATCH_SIZE * sizeof(float));
    hipMalloc((void**)&d_weightDeltas, NUM_LAYERS * NUM_NEURONS * NUM_NEURONS * sizeof(float));
    hipMalloc((void**)&d_biasDeltas, NUM_LAYERS * NUM_NEURONS * sizeof(float));

    // Copy data from host to device
    hipMemcpy(d_input, h_input, BATCH_SIZE * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_weights, h_weights, NUM_LAYERS * NUM_NEURONS * NUM_NEURONS * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_biases, h_biases, NUM_LAYERS * NUM_NEURONS * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_error, h_error, BATCH_SIZE * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_weightDeltas, h_weightDeltas, NUM_LAYERS * NUM_NEURONS * NUM_NEURONS * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_biasDeltas, h_biasDeltas, NUM_LAYERS * NUM_NEURONS * sizeof(float), hipMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 blockSize(NUM_NEURONS);
    dim3 gridSize(1);

    // Learning rate
    float learningRate = 0.01f;

    // Perform forward pass and backpropagation for each layer
    for (int layer = 0; layer < NUM_LAYERS; ++layer) {
        forwardPass<<<gridSize, blockSize>>>(d_input, d_weights + layer * NUM_NEURONS * NUM_NEURONS, d_biases + layer * NUM_NEURONS, d_output);
        hipDeviceSynchronize();

        backpropagation<<<gridSize, blockSize>>>(d_input, d_weights + layer * NUM_NEURONS * NUM_NEURONS, d_biases + layer * NUM_NEURONS, d_output, d_error, d_weightDeltas + layer * NUM_NEURONS * NUM_NEURONS, d_biasDeltas + layer * NUM_NEURONS, learningRate);
        hipDeviceSynchronize();

        // Update input for the next layer
        hipMemcpy(d_input, d_output, BATCH_SIZE * sizeof(float), hipMemcpyDeviceToDevice);
    }

    // Copy results back to host
    hipMemcpy(h_output, d_output, BATCH_SIZE * sizeof(float), hipMemcpyDeviceToHost);

    // Free device memory
    hipFree(d_input);
    hipFree(d_weights);
    hipFree(d_biases);
    hipFree(d_output);
    hipFree(d_error);
    hipFree(d_weightDeltas);
    hipFree(d_biasDeltas);

    // Free host memory
    free(h_input);
    free(h_weights);
    free(h_biases);
    free(h_output);
    free(h_error);
    free(h_weightDeltas);
    free(h_biasDeltas);

    return 0;
}