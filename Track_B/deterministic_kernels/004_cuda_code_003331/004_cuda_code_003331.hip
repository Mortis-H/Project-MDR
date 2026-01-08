#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

// Constants for the neural network
#define NUM_LAYERS 3
#define NUM_NEURONS 128
#define BATCH_SIZE 256
#define LEARNING_RATE 0.01f

// Kernel to initialize weights and biases with random values
__global__ void initializeParameters(float *weights, float *biases, int seed) {
    hiprandState state;
    hiprand_init(seed, blockIdx.x * blockDim.x + threadIdx.x, 0, &state);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_LAYERS * NUM_NEURONS * NUM_NEURONS) {
        weights[idx] = hiprand_uniform(&state) * 0.01f;
    }
    if (idx < NUM_LAYERS * NUM_NEURONS) {
        biases[idx] = hiprand_uniform(&state) * 0.01f;
    }
}

// Activation function (ReLU)
__device__ float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

// Derivative of ReLU
__device__ float reluDerivative(float x) {
    return x > 0.0f ? 1.0f : 0.0f;
}

// Forward pass kernel
__global__ void forwardPass(float *inputs, float *weights, float *biases, float *outputs, int batch_size) {
    int neuronIdx = threadIdx.x;
    int layerIdx = blockIdx.x;
    int batchIdx = blockIdx.y;

    if (neuronIdx < NUM_NEURONS) {
        float sum = 0.0f;
        for (int j = 0; j < NUM_NEURONS; j++) {
            sum += inputs[batchIdx * NUM_NEURONS + j] * weights[layerIdx * NUM_NEURONS * NUM_NEURONS + neuronIdx * NUM_NEURONS + j];
        }
        sum += biases[layerIdx * NUM_NEURONS + neuronIdx];
        outputs[batchIdx * NUM_NEURONS + neuronIdx] = relu(sum);
    }
}

// Backward pass kernel
__global__ void backwardPass(float *inputs, float *weights, float *biases, float *outputs, float *dOutputs, float *dWeights, float *dBiases, int batch_size) {
    int neuronIdx = threadIdx.x;
    int layerIdx = blockIdx.x;
    int batchIdx = blockIdx.y;

    if (neuronIdx < NUM_NEURONS) {
        float dSum = dOutputs[batchIdx * NUM_NEURONS + neuronIdx] * reluDerivative(outputs[batchIdx * NUM_NEURONS + neuronIdx]);
        for (int j = 0; j < NUM_NEURONS; j++) {
            atomicAdd(&dWeights[layerIdx * NUM_NEURONS * NUM_NEURONS + neuronIdx * NUM_NEURONS + j], dSum * inputs[batchIdx * NUM_NEURONS + j]);
        }
        atomicAdd(&dBiases[layerIdx * NUM_NEURONS + neuronIdx], dSum);
    }
}

// Update weights and biases
__global__ void updateParameters(float *weights, float *biases, float *dWeights, float *dBiases, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < NUM_LAYERS * NUM_NEURONS * NUM_NEURONS) {
        weights[idx] -= LEARNING_RATE * dWeights[idx] / batch_size;
        dWeights[idx] = 0.0f; // Reset gradient
    }
    if (idx < NUM_LAYERS * NUM_NEURONS) {
        biases[idx] -= LEARNING_RATE * dBiases[idx] / batch_size;
        dBiases[idx] = 0.0f; // Reset gradient
    }
}

int main() {
    // Allocate memory for inputs, weights, biases, outputs, and gradients
    float *h_inputs, *h_weights, *h_biases, *h_outputs, *h_dWeights, *h_dBiases;
    float *d_inputs, *d_weights, *d_biases, *d_outputs, *d_dWeights, *d_dBiases;

    // Allocate host memory
    h_inputs = (float *)malloc(BATCH_SIZE * NUM_NEURONS * sizeof(float));
    h_weights = (float *)malloc(NUM_LAYERS * NUM_NEURONS * NUM_NEURONS * sizeof(float));
    h_biases = (float *)malloc(NUM_LAYERS * NUM_NEURONS * sizeof(float));
    h_outputs = (float *)malloc(BATCH_SIZE * NUM_NEURONS * sizeof(float));
    h_dWeights = (float *)malloc(NUM_LAYERS * NUM_NEURONS * NUM_NEURONS * sizeof(float));
    h_dBiases = (float *)malloc(NUM_LAYERS * NUM_NEURONS * sizeof(float));

    // Allocate device memory
    hipMalloc(&d_inputs, BATCH_SIZE * NUM_NEURONS * sizeof(float));
    hipMalloc(&d_weights, NUM_LAYERS * NUM_NEURONS * NUM_NEURONS * sizeof(float));
    hipMalloc(&d_biases, NUM_LAYERS * NUM_NEURONS * sizeof(float));
    hipMalloc(&d_outputs, BATCH_SIZE * NUM_NEURONS * sizeof(float));
    hipMalloc(&d_dWeights, NUM_LAYERS * NUM_NEURONS * NUM_NEURONS * sizeof(float));
    hipMalloc(&d_dBiases, NUM_LAYERS * NUM_NEURONS * sizeof(float));

    // Initialize weights and biases
    initializeParameters<<<(NUM_LAYERS * NUM_NEURONS * NUM_NEURONS + 255) / 256, 256>>>(d_weights, d_biases, time(NULL));
    hipDeviceSynchronize();

    // Initialize gradients to zero
    hipMemset(d_dWeights, 0, NUM_LAYERS * NUM_NEURONS * NUM_NEURONS * sizeof(float));
    hipMemset(d_dBiases, 0, NUM_LAYERS * NUM_NEURONS * sizeof(float));

    // Simulate input data
    for (int i = 0; i < BATCH_SIZE * NUM_NEURONS; i++) {
        h_inputs[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    hipMemcpy(d_inputs, h_inputs, BATCH_SIZE * NUM_NEURONS * sizeof(float), hipMemcpyHostToDevice);

    // Perform forward pass
    dim3 grid(NUM_LAYERS, BATCH_SIZE);
    dim3 block(NUM_NEURONS);
    forwardPass<<<grid, block>>>(d_inputs, d_weights, d_biases, d_outputs, BATCH_SIZE);
    hipDeviceSynchronize();

    // Simulate output gradient (dOutputs)
    for (int i = 0; i < BATCH_SIZE * NUM_NEURONS; i++) {
        h_dBiases[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    hipMemcpy(d_dBiases, h_dBiases, BATCH_SIZE * NUM_NEURONS * sizeof(float), hipMemcpyHostToDevice);

    // Perform backward pass
    backwardPass<<<grid, block>>>(d_inputs, d_weights, d_biases, d_outputs, d_dBiases, d_dWeights, d_dBiases, BATCH_SIZE);
    hipDeviceSynchronize();

    // Update weights and biases
    updateParameters<<<(NUM_LAYERS * NUM_NEURONS * NUM_NEURONS + 255) / 256, 256>>>(d_weights, d_biases, d_dWeights, d_dBiases, BATCH_SIZE);
    hipDeviceSynchronize();

    // Free memory
    free(h_inputs);
    free(h_weights);
    free(h_biases);
    free(h_outputs);
    free(h_dWeights);
    free(h_dBiases);

    hipFree(d_inputs);
    hipFree(d_weights);
    hipFree(d_biases);
    hipFree(d_outputs);
    hipFree(d_dWeights);
    hipFree(d_dBiases);

    return 0;
}