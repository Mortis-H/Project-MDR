#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define N 512  // Number of neurons per layer
#define BATCH_SIZE 256  // Batch size for training
#define INPUT_SIZE 512  // Size of input vector
#define OUTPUT_SIZE 512  // Size of output vector
#define LEARNING_RATE 0.01f

__global__ void initialize_weights(float *weights, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        hiprandState state;
        hiprand_init(idx, 0, 0, &state);
        weights[idx] = hiprand_uniform(&state) * 0.01f;
    }
}

__global__ void forward_pass(float *input, float *weights1, float *weights2, float *output, float *hidden) {
    extern __shared__ float shared_data[];

    // Load weights into shared memory
    int idx = threadIdx.x;
    if (idx < N) {
        shared_data[idx] = weights1[blockIdx.x * N + idx];
        shared_data[idx + N] = weights2[blockIdx.x * N + idx];
    }
    __syncthreads();

    // Compute hidden layer
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        sum += input[blockIdx.x * INPUT_SIZE + i] * shared_data[i];
    }
    hidden[blockIdx.x] = tanh(sum);
    __syncthreads();

    // Compute output layer
    sum = 0.0f;
    for (int i = 0; i < N; i++) {
        sum += hidden[blockIdx.x] * shared_data[i + N];
    }
    output[blockIdx.x] = tanh(sum);
}

__global__ void backward_pass(float *input, float *weights1, float *weights2, float *output, float *hidden, float *target) {
    extern __shared__ float shared_data[];

    // Load weights into shared memory
    int idx = threadIdx.x;
    if (idx < N) {
        shared_data[idx] = weights1[blockIdx.x * N + idx];
        shared_data[idx + N] = weights2[blockIdx.x * N + idx];
    }
    __syncthreads();

    // Compute output error
    float output_error = (output[blockIdx.x] - target[blockIdx.x]) * (1 - output[blockIdx.x] * output[blockIdx.x]);
    float hidden_error = output_error * shared_data[idx + N] * (1 - hidden[blockIdx.x] * hidden[blockIdx.x]);

    // Update weights
    for (int i = 0; i < N; i++) {
        if (idx < N) {
            atomicAdd(&weights2[blockIdx.x * N + idx], -LEARNING_RATE * hidden[blockIdx.x] * output_error);
            atomicAdd(&weights1[blockIdx.x * N + idx], -LEARNING_RATE * input[blockIdx.x * INPUT_SIZE + idx] * hidden_error);
        }
    }
}

int main() {
    float *h_input, *h_weights1, *h_weights2, *h_output, *h_hidden, *h_target;
    float *d_input, *d_weights1, *d_weights2, *d_output, *d_hidden, *d_target;

    // Allocate host memory
    h_input = (float *)malloc(BATCH_SIZE * INPUT_SIZE * sizeof(float));
    h_weights1 = (float *)malloc(N * N * sizeof(float));
    h_weights2 = (float *)malloc(N * N * sizeof(float));
    h_output = (float *)malloc(BATCH_SIZE * sizeof(float));
    h_hidden = (float *)malloc(BATCH_SIZE * sizeof(float));
    h_target = (float *)malloc(BATCH_SIZE * sizeof(float));

    // Initialize host data
    for (int i = 0; i < BATCH_SIZE * INPUT_SIZE; i++) h_input[i] = 0.1f;
    for (int i = 0; i < BATCH_SIZE; i++) h_target[i] = 0.5f;

    // Allocate device memory
    hipMalloc((void **)&d_input, BATCH_SIZE * INPUT_SIZE * sizeof(float));
    hipMalloc((void **)&d_weights1, N * N * sizeof(float));
    hipMalloc((void **)&d_weights2, N * N * sizeof(float));
    hipMalloc((void **)&d_output, BATCH_SIZE * sizeof(float));
    hipMalloc((void **)&d_hidden, BATCH_SIZE * sizeof(float));
    hipMalloc((void **)&d_target, BATCH_SIZE * sizeof(float));

    // Initialize weights
    initialize_weights<<<(N * N + 255) / 256, 256>>>(d_weights1, N * N);
    initialize_weights<<<(N * N + 255) / 256, 256>>>(d_weights2, N * N);
    hipDeviceSynchronize();

    // Copy input and target to device
    hipMemcpy(d_input, h_input, BATCH_SIZE * INPUT_SIZE * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_target, h_target, BATCH_SIZE * sizeof(float), hipMemcpyHostToDevice);

    // Asynchronous memory copies and kernel launches
    hipStream_t stream1, stream2;
    hipStreamCreate(&stream1);
    hipStreamCreate(&stream2);

    for (int epoch = 0; epoch < 1000; epoch++) {
        hipMemcpyAsync(d_input, h_input, BATCH_SIZE * INPUT_SIZE * sizeof(float), hipMemcpyHostToDevice, stream1);
        hipMemcpyAsync(d_target, h_target, BATCH_SIZE * sizeof(float), hipMemcpyHostToDevice, stream2);

        forward_pass<<<BATCH_SIZE, N, 2 * N * sizeof(float), stream1>>>(d_input, d_weights1, d_weights2, d_output, d_hidden);
        backward_pass<<<BATCH_SIZE, N, 2 * N * sizeof(float), stream2>>>(d_input, d_weights1, d_weights2, d_output, d_hidden, d_target);

        hipMemcpyAsync(h_output, d_output, BATCH_SIZE * sizeof(float), hipMemcpyDeviceToHost, stream1);
        hipMemcpyAsync(h_hidden, d_hidden, BATCH_SIZE * sizeof(float), hipMemcpyDeviceToHost, stream2);

        hipStreamSynchronize(stream1);
        hipStreamSynchronize(stream2);
    }

    // Clean up
    free(h_input);
    free(h_weights1);
    free(h_weights2);
    free(h_output);
    free(h_hidden);
    free(h_target);

    hipFree(d_input);
    hipFree(d_weights1);
    hipFree(d_weights2);
    hipFree(d_output);
    hipFree(d_hidden);
    hipFree(d_target);

    hipStreamDestroy(stream1);
    hipStreamDestroy(stream2);

    return 0;
}