#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define LAYERS 10
#define NEURONS 512
#define BATCH_SIZE 256
#define THREADS_PER_BLOCK 256

__global__ void initialize_weights(float *weights, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        hiprandState state;
        hiprand_init(idx, 0, 0, &state);
        weights[idx] = hiprand_uniform(&state) * 0.01f;
    }
}

__global__ void forward_pass(float *input, float *weights, float *output) {
    extern __shared__ float shared_weights[];

    int neuron_idx = threadIdx.x;
    int batch_idx = blockIdx.x;

    // Load weights into shared memory to avoid bank conflicts
    for (int layer = 0; layer < LAYERS; ++layer) {
        int weight_idx = layer * NEURONS * NEURONS + neuron_idx;
        shared_weights[neuron_idx] = weights[weight_idx];
        __syncthreads();

        float sum = 0.0f;
        for (int prev_neuron = 0; prev_neuron < NEURONS; ++prev_neuron) {
            sum += input[batch_idx * NEURONS + prev_neuron] * shared_weights[prev_neuron];
        }
        __syncthreads();

        // Apply ReLU activation
        output[batch_idx * NEURONS + neuron_idx] = (sum > 0.0f) ? sum : 0.0f;
        __syncthreads();

        // Move to next layer
        if (layer < LAYERS - 1) {
            input[batch_idx * NEURONS + neuron_idx] = output[batch_idx * NEURONS + neuron_idx];
        }
    }
}

int main() {
    float *h_input, *h_weights, *h_output;
    float *d_input, *d_weights, *d_output;

    // Allocate host memory
    h_input = (float *)malloc(BATCH_SIZE * NEURONS * sizeof(float));
    h_weights = (float *)malloc(LAYERS * NEURONS * NEURONS * sizeof(float));
    h_output = (float *)malloc(BATCH_SIZE * NEURONS * sizeof(float));

    // Initialize input data
    for (int i = 0; i < BATCH_SIZE * NEURONS; ++i) {
        h_input[i] = 0.5f; // Example input
    }

    // Allocate device memory
    hipMalloc((void **)&d_input, BATCH_SIZE * NEURONS * sizeof(float));
    hipMalloc((void **)&d_weights, LAYERS * NEURONS * NEURONS * sizeof(float));
    hipMalloc((void **)&d_output, BATCH_SIZE * NEURONS * sizeof(float));

    // Copy input data to device
    hipMemcpy(d_input, h_input, BATCH_SIZE * NEURONS * sizeof(float), hipMemcpyHostToDevice);

    // Initialize weights
    initialize_weights<<<(LAYERS * NEURONS * NEURONS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_weights, LAYERS * NEURONS * NEURONS);
    hipDeviceSynchronize();

    // Perform forward pass
    forward_pass<<<BATCH_SIZE, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(float)>>>(d_input, d_weights, d_output);
    hipDeviceSynchronize();

    // Copy output data to host
    hipMemcpy(h_output, d_output, BATCH_SIZE * NEURONS * sizeof(float), hipMemcpyDeviceToHost);

    // Free device memory
    hipFree(d_input);
    hipFree(d_weights);
    hipFree(d_output);

    // Free host memory
    free(h_input);
    free(h_weights);
    free(h_output);

    // Check for any errors launching the kernel
    hipError_t cudaStatus = hipGetLastError();
    if (cudaStatus != hipSuccess) {
        fprintf(stderr, "forward_pass launch failed: %s\n", hipGetErrorString(cudaStatus));
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight Systems or Nsight Compute to show complete traces.
    cudaStatus = hipDeviceReset();
    if (cudaStatus != hipSuccess) {
        fprintf(stderr, "hipDeviceReset failed!");
        return 1;
    }

    return 0;
}