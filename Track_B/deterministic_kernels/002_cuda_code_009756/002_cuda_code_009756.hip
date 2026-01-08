#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define NUM_LAYERS 10
#define NUM_NEURONS 512
#define BATCH_SIZE 256

__global__ void initializeWeights(float* weights, int numWeights) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numWeights) {
        hiprandState state;
        hiprand_init(idx, 0, 0, &state);
        weights[idx] = hiprand_normal(&state) * 0.01f;
    }
}

__global__ void forwardPass(float* input, float* weights, float* activations, int numLayers, int numNeurons, int batchSize) {
    extern __shared__ float sharedMem[];

    int neuronIdx = threadIdx.x;
    int batchIdx = blockIdx.x;

    for (int layer = 0; layer < numLayers; ++layer) {
        float sum = 0.0f;
        for (int i = 0; i < numNeurons; ++i) {
            int weightIdx = layer * numNeurons * numNeurons + neuronIdx * numNeurons + i;
            int inputIdx = (layer == 0) ? batchIdx * numNeurons + i : (layer - 1) * batchSize * numNeurons + batchIdx * numNeurons + i;
            sum += weights[weightIdx] * activations[inputIdx];
        }
        sharedMem[neuronIdx] = sum;
        __syncthreads();

        // Apply ReLU activation
        if (neuronIdx < numNeurons) {
            int outputIdx = layer * batchSize * numNeurons + batchIdx * numNeurons + neuronIdx;
            activations[outputIdx] = (sum > 0.0f) ? sum : 0.0f;
        }
        __syncthreads();
    }
}

int main() {
    float* d_weights, *d_activations;
    int numWeights = NUM_LAYERS * NUM_NEURONS * NUM_NEURONS;
    int numActivations = (NUM_LAYERS + 1) * BATCH_SIZE * NUM_NEURONS;

    hipError_t err;

    // Allocate memory on the device
    err = hipMalloc((void**)&d_weights, numWeights * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for weights" << std::endl;
        return -1;
    }

    err = hipMalloc((void**)&d_activations, numActivations * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for activations" << std::endl;
        hipFree(d_weights);
        return -1;
    }

    // Initialize weights
    int threadsPerBlock = 256;
    int blocksPerGrid = (numWeights + threadsPerBlock - 1) / threadsPerBlock;
    initializeWeights<<<blocksPerGrid, threadsPerBlock>>>(d_weights, numWeights);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel launch failed for initializeWeights" << std::endl;
        hipFree(d_weights);
        hipFree(d_activations);
        return -1;
    }

    // Simulate input activations
    float* h_input = new float[BATCH_SIZE * NUM_NEURONS];
    for (int i = 0; i < BATCH_SIZE * NUM_NEURONS; ++i) {
        h_input[i] = static_cast<float>(i % 2);
    }

    // Copy input to device
    err = hipMemcpy(d_activations, h_input, BATCH_SIZE * NUM_NEURONS * sizeof(float), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy input data to device" << std::endl;
        hipFree(d_weights);
        hipFree(d_activations);
        delete[] h_input;
        return -1;
    }

    // Perform forward pass
    threadsPerBlock = NUM_NEURONS;
    blocksPerGrid = BATCH_SIZE;
    forwardPass<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_activations, d_weights, d_activations, NUM_LAYERS, NUM_NEURONS, BATCH_SIZE);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel launch failed for forwardPass" << std::endl;
        hipFree(d_weights);
        hipFree(d_activations);
        delete[] h_input;
        return -1;
    }

    // Free memory
    hipFree(d_weights);
    hipFree(d_activations);
    delete[] h_input;

    return 0;
}