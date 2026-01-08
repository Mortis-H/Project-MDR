#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define LAYERS 5
#define NEURONS 128
#define BATCH_SIZE 256

__global__ void initialize_weights(float* weights, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        hiprandState state;
        hiprand_init((unsigned long long) clock() + idx, 0, 0, &state);
        weights[idx] = hiprand_uniform(&state) * 0.01f;
    }
}

__global__ void forward_pass(float* input, float* weights, float* output, int neurons, int layers, int batch_size) {
    extern __shared__ float shared_mem[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int neuron_idx = threadIdx.x % neurons;
    int layer = threadIdx.x / neurons;

    if (idx < batch_size * neurons * layers) {
        float sum = 0.0f;
        for (int i = 0; i < neurons; ++i) {
            sum += input[(layer * neurons + i) * batch_size + idx / neurons] * weights[(layer * neurons + i) * neurons + neuron_idx];
        }
        output[(layer * neurons + neuron_idx) * batch_size + idx / neurons] = tanh(sum);
    }
}

__global__ void backward_pass(float* input, float* weights, float* output, float* delta, int neurons, int layers, int batch_size) {
    extern __shared__ float shared_mem[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int neuron_idx = threadIdx.x % neurons;
    int layer = threadIdx.x / neurons;

    if (idx < batch_size * neurons * layers) {
        float sum = 0.0f;
        for (int i = 0; i < neurons; ++i) {
            sum += weights[(layer * neurons + neuron_idx) * neurons + i] * delta[(layer * neurons + i) * batch_size + idx / neurons];
        }
        delta[(layer * neurons + neuron_idx) * batch_size + idx / neurons] = sum * (1.0f - output[(layer * neurons + neuron_idx) * batch_size + idx / neurons] * output[(layer * neurons + neuron_idx) * batch_size + idx / neurons]);
    }
}

__global__ void update_weights(float* weights, float* input, float* delta, int neurons, int layers, int batch_size, float learning_rate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int neuron_idx = threadIdx.x % neurons;
    int layer = threadIdx.x / neurons;

    if (idx < batch_size * neurons * layers) {
        atomicAdd(&weights[(layer * neurons + neuron_idx) * neurons + neuron_idx], -learning_rate * input[(layer * neurons + neuron_idx) * batch_size + idx / neurons] * delta[(layer * neurons + neuron_idx) * batch_size + idx / neurons]);
    }
}

int main() {
    float* d_input, *d_weights, *d_output, *d_delta;
    float* h_input = new float[NEURONS * BATCH_SIZE];
    float* h_weights = new float[LAYERS * NEURONS * NEURONS];
    float* h_output = new float[LAYERS * NEURONS * BATCH_SIZE];
    float* h_delta = new float[LAYERS * NEURONS * BATCH_SIZE];

    int size_input = NEURONS * BATCH_SIZE * sizeof(float);
    int size_weights = LAYERS * NEURONS * NEURONS * sizeof(float);
    int size_output = LAYERS * NEURONS * BATCH_SIZE * sizeof(float);
    int size_delta = LAYERS * NEURONS * BATCH_SIZE * sizeof(float);

    hipError_t err;

    err = hipMalloc((void**)&d_input, size_input);
    if (err != hipSuccess) { std::cerr << "Error allocating d_input: " << hipGetErrorString(err) << std::endl; return -1; }
    err = hipMalloc((void**)&d_weights, size_weights);
    if (err != hipSuccess) { std::cerr << "Error allocating d_weights: " << hipGetErrorString(err) << std::endl; return -1; }
    err = hipMalloc((void**)&d_output, size_output);
    if (err != hipSuccess) { std::cerr << "Error allocating d_output: " << hipGetErrorString(err) << std::endl; return -1; }
    err = hipMalloc((void**)&d_delta, size_delta);
    if (err != hipSuccess) { std::cerr << "Error allocating d_delta: " << hipGetErrorString(err) << std::endl; return -1; }

    for (int i = 0; i < NEURONS * BATCH_SIZE; ++i) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    hipMemcpy(d_input, h_input, size_input, hipMemcpyHostToDevice);

    int threadsPerBlock = NEURONS * LAYERS;
    int blocksPerGrid = (BATCH_SIZE + threadsPerBlock - 1) / threadsPerBlock;

    hipStream_t stream;
    hipStreamCreate(&stream);

    initialize_weights<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_weights, LAYERS * NEURONS * NEURONS);
    hipDeviceSynchronize();

    float learning_rate = 0.01f;
    for (int epoch = 0; epoch < 100; ++epoch) {
        hipMemcpyAsync(d_output, d_input, size_input, hipMemcpyDeviceToDevice, stream);
        forward_pass<<<blocksPerGrid, threadsPerBlock, NEURONS * sizeof(float), stream>>>(d_output, d_weights, d_output, NEURONS, LAYERS, BATCH_SIZE);
        backward_pass<<<blocksPerGrid, threadsPerBlock, NEURONS * sizeof(float), stream>>>(d_output, d_weights, d_output, d_delta, NEURONS, LAYERS, BATCH_SIZE);
        update_weights<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_weights, d_output, d_delta, NEURONS, LAYERS, BATCH_SIZE, learning_rate);
        hipDeviceSynchronize();
    }

    hipMemcpy(h_output, d_output, size_output, hipMemcpyDeviceToHost);
    hipMemcpy(h_delta, d_delta, size_delta, hipMemcpyDeviceToHost);

    hipFree(d_input);
    hipFree(d_weights);
    hipFree(d_output);
    hipFree(d_delta);

    delete[] h_input;
    delete[] h_weights;
    delete[] h_output;
    delete[] h_delta;

    return 0;
}