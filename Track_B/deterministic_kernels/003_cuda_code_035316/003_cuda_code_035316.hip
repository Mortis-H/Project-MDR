#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define LAYERS 12
#define NEURONS 2048
#define BATCH_SIZE 256

__global__ void initialize_weights(float *weights, int num_weights) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_weights) {
        hiprandState state;
        hiprand_init(idx, 0, 0, &state);
        weights[idx] = hiprand_uniform(&state) * 0.1f;
    }
}

__global__ void forward_pass(float *input, float *weights, float *output, int layer) {
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_idx < NEURONS) {
        float sum = 0.0f;
        for (int i = 0; i < NEURONS; i++) {
            sum += input[i] * weights[layer * NEURONS * NEURONS + neuron_idx * NEURONS + i];
        }
        output[neuron_idx] = tanh(sum);
    }
}

__global__ void backward_pass(float *output, float *weights, float *d_weights, float *d_output, int layer) {
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_idx < NEURONS) {
        float delta = d_output[neuron_idx] * (1.0f - output[neuron_idx] * output[neuron_idx]);
        for (int i = 0; i < NEURONS; i++) {
            atomicAdd(&d_weights[layer * NEURONS * NEURONS + neuron_idx * NEURONS + i], delta * output[i]);
        }
        d_output[neuron_idx] = delta;
    }
}

void checkCudaError(hipError_t err, const char *msg) {
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << msg << " (" << hipGetErrorString(err) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    float *h_input, *h_weights, *h_output, *h_d_weights, *h_d_output;
    float *d_input, *d_weights, *d_output, *d_d_weights, *d_d_output;

    // Allocate host memory
    h_input = (float *)malloc(BATCH_SIZE * NEURONS * sizeof(float));
    h_weights = (float *)malloc(LAYERS * NEURONS * NEURONS * sizeof(float));
    h_output = (float *)malloc(BATCH_SIZE * NEURONS * sizeof(float));
    h_d_weights = (float *)malloc(LAYERS * NEURONS * NEURONS * sizeof(float));
    h_d_output = (float *)malloc(BATCH_SIZE * NEURONS * sizeof(float));

    // Allocate device memory
    checkCudaError(hipMalloc((void **)&d_input, BATCH_SIZE * NEURONS * sizeof(float)), "hipMalloc d_input");
    checkCudaError(hipMalloc((void **)&d_weights, LAYERS * NEURONS * NEURONS * sizeof(float)), "hipMalloc d_weights");
    checkCudaError(hipMalloc((void **)&d_output, BATCH_SIZE * NEURONS * sizeof(float)), "hipMalloc d_output");
    checkCudaError(hipMalloc((void **)&d_d_weights, LAYERS * NEURONS * NEURONS * sizeof(float)), "hipMalloc d_d_weights");
    checkCudaError(hipMalloc((void **)&d_d_output, BATCH_SIZE * NEURONS * sizeof(float)), "hipMalloc d_d_output");

    // Initialize host memory
    for (int i = 0; i < BATCH_SIZE * NEURONS; i++) {
        h_input[i] = (float)rand() / RAND_MAX;
        h_output[i] = 0.0f;
        h_d_output[i] = 0.0f;
    }

    // Initialize weights on device
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((NEURONS * NEURONS) / threadsPerBlock.x + 1);
    initialize_weights<<<blocksPerGrid, threadsPerBlock>>>(d_weights, LAYERS * NEURONS * NEURONS);
    checkCudaError(hipGetLastError(), "initialize_weights kernel launch failed");

    // Asynchronous memory copies and kernel launches
    hipStream_t stream;
    checkCudaError(hipStreamCreate(&stream), "hipStreamCreate");

    for (int epoch = 0; epoch < 10; epoch++) {
        // Forward pass
        for (int layer = 0; layer < LAYERS; layer++) {
            if (layer == 0) {
                hipMemcpyAsync(d_input, h_input, BATCH_SIZE * NEURONS * sizeof(float), hipMemcpyHostToDevice, stream);
            } else {
                hipMemcpyAsync(d_input, d_output, BATCH_SIZE * NEURONS * sizeof(float), hipMemcpyDeviceToDevice, stream);
            }
            forward_pass<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_input, d_weights, d_output, layer);
            checkCudaError(hipGetLastError(), "forward_pass kernel launch failed");
        }

        // Backward pass
        for (int layer = LAYERS - 1; layer >= 0; layer--) {
            if (layer == LAYERS - 1) {
                hipMemcpyAsync(d_d_output, h_d_output, BATCH_SIZE * NEURONS * sizeof(float), hipMemcpyHostToDevice, stream);
            }
            backward_pass<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_output, d_weights, d_d_weights, d_d_output, layer);
            checkCudaError(hipGetLastError(), "backward_pass kernel launch failed");
        }

        // Synchronize stream to ensure all operations are complete
        checkCudaError(hipStreamSynchronize(stream), "hipStreamSynchronize");
    }

    // Clean up
    free(h_input);
    free(h_weights);
    free(h_output);
    free(h_d_weights);
    free(h_d_output);
    hipFree(d_input);
    hipFree(d_weights);
    hipFree(d_output);
    hipFree(d_d_weights);
    hipFree(d_d_output);
    hipStreamDestroy(stream);

    return 0;
}