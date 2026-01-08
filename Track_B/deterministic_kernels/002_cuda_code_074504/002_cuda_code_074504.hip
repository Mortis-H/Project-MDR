#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define N 512
#define BLOCK_SIZE 16

__global__ void initialize_weights(float *weights, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        hiprandState s;
        hiprand_init(idx, 0, 0, &s);
        weights[idx] = hiprand_uniform(&s) * 0.01f; // Initialize weights with small random values
    }
}

__global__ void forward_pass(float *input, float *weights1, float *weights2, float *output, int batch_size) {
    extern __shared__ float shared_data[];

    int bx = blockIdx.x;
    int tx = threadIdx.x;

    // Load input and weights into shared memory to avoid bank conflicts
    if (tx < N) {
        shared_data[tx] = input[bx * N + tx];
        shared_data[N + tx] = weights1[tx * N + tx];
        shared_data[2 * N + tx] = weights2[tx * N + tx];
    }
    __syncthreads();

    float sum1 = 0.0f;
    float sum2 = 0.0f;

    // Perform matrix-vector multiplication for the first layer
    for (int i = 0; i < N; i += BLOCK_SIZE) {
        int idx = i + tx;
        if (idx < N) {
            sum1 += shared_data[idx] * shared_data[N + idx];
        }
        __syncthreads();
    }

    // Perform matrix-vector multiplication for the second layer
    for (int i = 0; i < N; i += BLOCK_SIZE) {
        int idx = i + tx;
        if (idx < N) {
            sum2 += sum1 * shared_data[2 * N + idx];
        }
        __syncthreads();
    }

    // Store the result in global memory
    if (tx < N) {
        output[bx * N + tx] = sum2;
    }
}

int main() {
    int batch_size = 1024;
    int input_size = batch_size * N;
    int weights_size = N * N;

    float *h_input, *h_weights1, *h_weights2, *h_output;
    float *d_input, *d_weights1, *d_weights2, *d_output;

    // Allocate host memory
    h_input = (float *)malloc(input_size * sizeof(float));
    h_weights1 = (float *)malloc(weights_size * sizeof(float));
    h_weights2 = (float *)malloc(weights_size * sizeof(float));
    h_output = (float *)malloc(input_size * sizeof(float));

    // Initialize input with random values
    for (int i = 0; i < input_size; i++) {
        h_input[i] = (float)(rand() % 100) / 100.0f;
    }

    // Allocate device memory
    hipError_t err = hipMalloc((void **)&d_input, input_size * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device input memory: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc((void **)&d_weights1, weights_size * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device weights1 memory: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc((void **)&d_weights2, weights_size * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device weights2 memory: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc((void **)&d_output, input_size * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device output memory: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy input to device
    err = hipMemcpy(d_input, h_input, input_size * sizeof(float), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy input to device: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Initialize weights on the device
    initialize_weights<<<(weights_size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_weights1, weights_size);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to initialize weights1: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    hipDeviceSynchronize();

    initialize_weights<<<(weights_size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_weights2, weights_size);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to initialize weights2: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    hipDeviceSynchronize();

    // Launch forward pass kernel
    forward_pass<<<batch_size / BLOCK_SIZE, BLOCK_SIZE, 3 * N * sizeof(float)>>>(d_input, d_weights1, d_weights2, d_output, batch_size);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch forward pass kernel: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    hipDeviceSynchronize();

    // Copy output back to host
    err = hipMemcpy(h_output, d_output, input_size * sizeof(float), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy output to host: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Free device memory
    hipFree(d_input);
    hipFree(d_weights1);
    hipFree(d_weights2);
    hipFree(d_output);

    // Free host memory
    free(h_input);
    free(h_weights1);
    free(h_weights2);
    free(h_output);

    std::cout << "CUDA forward pass completed successfully." << std::endl;
    return 0;
}