#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define LAYERS 10
#define NEURONS 512
#define BATCH_SIZE 256
#define LEARNING_RATE 0.01f

__global__ void initializeWeights(float *weights, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        hiprandState state;
        hiprand_init((unsigned long long)clock() + idx, 0, 0, &state);
        weights[idx] = hiprand_uniform(&state) * 0.01f;
    }
}

__global__ void forwardPass(float *inputs, float *weights, float *outputs, int numInputs) {
    int neuronIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuronIdx < NEURONS) {
        float sum = 0.0f;
        for (int i = 0; i < numInputs; ++i) {
            sum += inputs[i] * weights[neuronIdx + i * NEURONS];
        }
        outputs[neuronIdx] = 1.0f / (1.0f + expf(-sum)); // Sigmoid activation
    }
}

__global__ void computeGradients(float *outputs, float *expected, float *gradients, int layer) {
    int neuronIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuronIdx < NEURONS) {
        float error = outputs[neuronIdx] - expected[neuronIdx];
        float gradient = error * outputs[neuronIdx] * (1.0f - outputs[neuronIdx]);
        atomicAdd(&gradients[neuronIdx + layer * NEURONS], gradient);
    }
}

__global__ void updateWeights(float *weights, float *gradients, int layer) {
    int neuronIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuronIdx < NEURONS) {
        atomicAdd(&weights[neuronIdx + layer * NEURONS], -LEARNING_RATE * gradients[neuronIdx + layer * NEURONS]);
    }
}

void checkCudaError(hipError_t err, const char *operation) {
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << operation << " failed with error " << hipGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    float *h_inputs = new float[NEURONS];
    float *h_expected = new float[NEURONS];
    float *h_weights = new float[LAYERS * NEURONS * NEURONS];
    float *h_gradients = new float[LAYERS * NEURONS];

    // Initialize inputs and expected outputs
    for (int i = 0; i < NEURONS; ++i) {
        h_inputs[i] = static_cast<float>(rand()) / RAND_MAX;
        h_expected[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float *d_inputs, *d_expected, *d_weights, *d_outputs[LAYERS], *d_gradients;
    hipError_t err;

    err = hipMalloc(&d_inputs, NEURONS * sizeof(float));
    checkCudaError(err, "hipMalloc d_inputs");
    err = hipMalloc(&d_expected, NEURONS * sizeof(float));
    checkCudaError(err, "hipMalloc d_expected");
    err = hipMalloc(&d_weights, LAYERS * NEURONS * NEURONS * sizeof(float));
    checkCudaError(err, "hipMalloc d_weights");
    err = hipMalloc(&d_gradients, LAYERS * NEURONS * sizeof(float));
    checkCudaError(err, "hipMalloc d_gradients");

    for (int i = 0; i < LAYERS; ++i) {
        err = hipMalloc(&d_outputs[i], NEURONS * sizeof(float));
        checkCudaError(err, "hipMalloc d_outputs");
    }

    err = hipMemcpy(d_inputs, h_inputs, NEURONS * sizeof(float), hipMemcpyHostToDevice);
    checkCudaError(err, "hipMemcpy d_inputs");
    err = hipMemcpy(d_expected, h_expected, NEURONS * sizeof(float), hipMemcpyHostToDevice);
    checkCudaError(err, "hipMemcpy d_expected");

    int blockSize = 256;
    int numBlocks = (NEURONS + blockSize - 1) / blockSize;

    // Initialize weights
    initializeWeights<<<numBlocks, blockSize>>>(d_weights, LAYERS * NEURONS * NEURONS);
    err = hipGetLastError();
    checkCudaError(err, "initializeWeights");

    // Training loop
    for (int epoch = 0; epoch < 1000; ++epoch) {
        float *prevOutput = d_inputs;
        for (int layer = 0; layer < LAYERS; ++layer) {
            forwardPass<<<numBlocks, blockSize>>>(prevOutput, d_weights + layer * NEURONS * NEURONS, d_outputs[layer], layer == 0 ? NEURONS : NEURONS);
            err = hipGetLastError();
            checkCudaError(err, "forwardPass");

            prevOutput = d_outputs[layer];
        }

        // Compute gradients
        computeGradients<<<numBlocks, blockSize>>>(d_outputs[LAYERS - 1], d_expected, d_gradients, LAYERS - 1);
        err = hipGetLastError();
        checkCudaError(err, "computeGradients");

        // Update weights
        updateWeights<<<numBlocks, blockSize>>>(d_weights + (LAYERS - 1) * NEURONS * NEURONS, d_gradients, LAYERS - 1);
        err = hipGetLastError();
        checkCudaError(err, "updateWeights");
    }

    // Free device memory
    err = hipFree(d_inputs);
    checkCudaError(err, "hipFree d_inputs");
    err = hipFree(d_expected);
    checkCudaError(err, "hipFree d_expected");
    err = hipFree(d_weights);
    checkCudaError(err, "hipFree d_weights");
    err = hipFree(d_gradients);
    checkCudaError(err, "hipFree d_gradients");

    for (int i = 0; i < LAYERS; ++i) {
        err = hipFree(d_outputs[i]);
        checkCudaError(err, "hipFree d_outputs");
    }

    // Free host memory
    delete[] h_inputs;
    delete[] h_expected;
    delete[] h_weights;
    delete[] h_gradients;

    std::cout << "Training complete." << std::endl;
    return 0;
}