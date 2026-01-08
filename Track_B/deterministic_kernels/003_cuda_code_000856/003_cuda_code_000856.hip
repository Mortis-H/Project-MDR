#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <hiprand/hiprand_kernel.h>

namespace cg = cooperative_groups;

#define LAYERS 5
#define NEURONS 64
#define BATCH_SIZE 256
#define LEARNING_RATE 0.01f

__global__ void initializeWeights(float* weights, hiprandState* states, int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < LAYERS * NEURONS * NEURONS) {
        hiprandState localState = states[idx];
        hiprand_init(seed, idx, 0, &localState);
        weights[idx] = hiprand_normal(&localState) * 0.01f;
        states[idx] = localState;
    }
}

__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ float sigmoidDerivative(float x) {
    float s = sigmoid(x);
    return s * (1.0f - s);
}

__global__ void forwardPass(float* inputs, float* weights, float* activations, int* indices) {
    extern __shared__ float sharedMemory[];
    float* sharedActivations = sharedMemory;
    float* sharedWeights = sharedMemory + NEURONS;

    int layer = indices[0];
    int neuron = threadIdx.x;
    int batch = blockIdx.x;

    // Load weights into shared memory
    if (neuron < NEURONS) {
        for (int i = 0; i < NEURONS; ++i) {
            sharedWeights[neuron * NEURONS + i] = weights[layer * NEURONS * NEURONS + neuron * NEURONS + i];
        }
    }
    __syncthreads();

    // Compute activations for each neuron
    float sum = 0.0f;
    if (neuron < NEURONS) {
        for (int i = 0; i < NEURONS; ++i) {
            sum += sharedWeights[neuron * NEURONS + i] * inputs[batch * NEURONS + i];
        }
        sharedActivations[neuron] = sigmoid(sum);
    }
    __syncthreads();

    // Store activations back to global memory
    if (neuron < NEURONS) {
        activations[batch * NEURONS + neuron] = sharedActivations[neuron];
    }
}

__global__ void backwardPass(float* inputs, float* weights, float* activations, float* deltas, float* errors, int* indices) {
    extern __shared__ float sharedMemory[];
    float* sharedActivations = sharedMemory;
    float* sharedWeights = sharedMemory + NEURONS;
    float* sharedDeltas = sharedMemory + 2 * NEURONS;

    int layer = indices[0];
    int neuron = threadIdx.x;
    int batch = blockIdx.x;

    // Load weights and activations into shared memory
    if (neuron < NEURONS) {
        for (int i = 0; i < NEURONS; ++i) {
            sharedWeights[neuron * NEURONS + i] = weights[layer * NEURONS * NEURONS + neuron * NEURONS + i];
        }
        sharedActivations[neuron] = activations[batch * NEURONS + neuron];
    }
    __syncthreads();

    // Compute deltas for each neuron
    float delta = 0.0f;
    if (neuron < NEURONS) {
        if (layer == LAYERS - 1) {
            delta = (inputs[batch * NEURONS + neuron] - sharedActivations[neuron]) * sigmoidDerivative(sharedActivations[neuron]);
        } else {
            for (int i = 0; i < NEURONS; ++i) {
                delta += sharedWeights[i * NEURONS + neuron] * sharedDeltas[i];
            }
            delta *= sigmoidDerivative(sharedActivations[neuron]);
        }
        sharedDeltas[neuron] = delta;
    }
    __syncthreads();

    // Update weights
    if (neuron < NEURONS) {
        for (int i = 0; i < NEURONS; ++i) {
            float* prevActivations = (layer == 0) ? inputs : activations + (batch * NEURONS);
            weights[layer * NEURONS * NEURONS + neuron * NEURONS + i] += LEARNING_RATE * delta * prevActivations[i * NEURONS + i];
        }
    }

    // Store errors back to global memory
    if (neuron < NEURONS) {
        errors[batch * NEURONS + neuron] = delta;
    }
}

int main() {
    float* h_inputs = new float[BATCH_SIZE * NEURONS];
    float* h_weights = new float[LAYERS * NEURONS * NEURONS];
    float* h_activations = new float[BATCH_SIZE * NEURONS];
    float* h_errors = new float[BATCH_SIZE * NEURONS];
    float* h_deltas = new float[BATCH_SIZE * NEURONS];

    // Initialize inputs with random values
    for (int i = 0; i < BATCH_SIZE * NEURONS; ++i) {
        h_inputs[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate device memory
    float* d_inputs, *d_weights, *d_activations, *d_errors, *d_deltas;
    hipMalloc(&d_inputs, BATCH_SIZE * NEURONS * sizeof(float));
    hipMalloc(&d_weights, LAYERS * NEURONS * NEURONS * sizeof(float));
    hipMalloc(&d_activations, BATCH_SIZE * NEURONS * sizeof(float));
    hipMalloc(&d_errors, BATCH_SIZE * NEURONS * sizeof(float));
    hipMalloc(&d_deltas, BATCH_SIZE * NEURONS * sizeof(float));

    // Allocate and initialize random states
    hiprandState* d_states;
    hipMalloc(&d_states, LAYERS * NEURONS * NEURONS * sizeof(hiprandState));
    initializeWeights<<<(LAYERS * NEURONS * NEURONS + 255) / 256, 256>>>(d_weights, d_states, 1234);

    // Copy inputs to device
    hipMemcpy(d_inputs, h_inputs, BATCH_SIZE * NEURONS * sizeof(float), hipMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 blockSize(NEURONS);
    dim3 gridSize(BATCH_SIZE);

    // Define indices for layers
    int h_indices[1];
    int* d_indices;
    hipMalloc(&d_indices, sizeof(int));
    hipMemcpy(d_indices, h_indices, sizeof(int), hipMemcpyHostToDevice);

    // Forward and backward passes
    for (int epoch = 0; epoch < 1000; ++epoch) {
        for (int layer = 0; layer < LAYERS; ++layer) {
            h_indices[0] = layer;
            hipMemcpy(d_indices, h_indices, sizeof(int), hipMemcpyHostToDevice);

            // Forward pass
            forwardPass<<<gridSize, blockSize, NEURONS * sizeof(float) * 3>>>(d_inputs, d_weights, d_activations, d_indices);

            // Backward pass
            backwardPass<<<gridSize, blockSize, NEURONS * sizeof(float) * 3>>>(d_inputs, d_weights, d_activations, d_deltas, d_errors, d_indices);
        }
    }

    // Copy results back to host
    hipMemcpy(h_activations, d_activations, BATCH_SIZE * NEURONS * sizeof(float), hipMemcpyDeviceToHost);

    // Free device memory
    hipFree(d_inputs);
    hipFree(d_weights);
    hipFree(d_activations);
    hipFree(d_errors);
    hipFree(d_deltas);
    hipFree(d_states);
    hipFree(d_indices);

    // Free host memory
    delete[] h_inputs;
    delete[] h_weights;
    delete[] h_activations;
    delete[] h_errors;
    delete[] h_deltas;

    // Check for any errors launching the kernel
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    std::cout << "Training completed successfully." << std::endl;
    return 0;
}