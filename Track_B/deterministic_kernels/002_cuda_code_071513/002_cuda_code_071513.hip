#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define NUM_LAYERS 4
#define NUM_NEURONS 64
#define BATCH_SIZE 256
#define LEARNING_RATE 0.01f

__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ float sigmoid_derivative(float x) {
    float s = sigmoid(x);
    return s * (1.0f - s);
}

__global__ void initialize_weights(float *weights, int num_weights) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_weights) {
        hiprandState s;
        hiprand_init((unsigned long long) clock() + idx, 0, 0, &s);
        weights[idx] = hiprand_uniform(&s) * 2.0f - 1.0f;
    }
}

__global__ void forward_pass(float *input, float *weights, float *outputs, int layer) {
    extern __shared__ float shared_data[];
    int neuron = threadIdx.x;
    float sum = 0.0f;

    for (int i = 0; i < NUM_NEURONS; i++) {
        sum += input[i] * weights[layer * NUM_NEURONS * NUM_NEURONS + neuron * NUM_NEURONS + i];
    }

    outputs[neuron] = sigmoid(sum);
}

__global__ void compute_output_error(float *output, float *target, float *output_error) {
    int neuron = threadIdx.x;
    output_error[neuron] = (output[neuron] - target[neuron]) * sigmoid_derivative(output[neuron]);
}

__global__ void backpropagation(float *weights, float *outputs, float *errors, float *next_errors, int layer) {
    extern __shared__ float shared_data[];
    int neuron = threadIdx.x;
    float sum = 0.0f;

    for (int i = 0; i < NUM_NEURONS; i++) {
        sum += next_errors[i] * weights[(layer + 1) * NUM_NEURONS * NUM_NEURONS + i * NUM_NEURONS + neuron];
    }

    errors[neuron] = sum * sigmoid_derivative(outputs[neuron]);
}

__global__ void update_weights(float *weights, float *outputs, float *errors, int layer) {
    int neuron = threadIdx.x;
    for (int i = 0; i < NUM_NEURONS; i++) {
        atomicAdd(&weights[layer * NUM_NEURONS * NUM_NEURONS + neuron * NUM_NEURONS + i],
                  -LEARNING_RATE * errors[neuron] * outputs[i]);
    }
}

void train_neural_network(float *d_input, float *d_target, float *d_weights, int num_batches) {
    float *d_outputs[NUM_LAYERS];
    float *d_errors[NUM_LAYERS];
    hipMalloc(&d_outputs[0], NUM_NEURONS * sizeof(float));
    hipMalloc(&d_errors[NUM_LAYERS - 1], NUM_NEURONS * sizeof(float));

    for (int i = 1; i < NUM_LAYERS; i++) {
        hipMalloc(&d_outputs[i], NUM_NEURONS * sizeof(float));
        hipMalloc(&d_errors[i - 1], NUM_NEURONS * sizeof(float));
    }

    dim3 blockSize(NUM_NEURONS);
    dim3 gridSize(1);

    for (int batch = 0; batch < num_batches; batch++) {
        // Forward pass
        for (int layer = 0; layer < NUM_LAYERS; layer++) {
            forward_pass<<<gridSize, blockSize, NUM_NEURONS * sizeof(float)>>>(layer == 0 ? d_input : d_outputs[layer - 1], d_weights, d_outputs[layer], layer);
            hipDeviceSynchronize();
        }

        // Compute output error
        compute_output_error<<<gridSize, blockSize>>>(d_outputs[NUM_LAYERS - 1], d_target, d_errors[NUM_LAYERS - 2]);
        hipDeviceSynchronize();

        // Backpropagation
        for (int layer = NUM_LAYERS - 2; layer >= 0; layer--) {
            backpropagation<<<gridSize, blockSize, NUM_NEURONS * sizeof(float)>>>(d_weights, d_outputs[layer], d_errors[layer], layer == NUM_LAYERS - 2 ? d_errors[NUM_LAYERS - 2] : d_errors[layer + 1], layer);
            hipDeviceSynchronize();
        }

        // Update weights
        for (int layer = 0; layer < NUM_LAYERS - 1; layer++) {
            update_weights<<<gridSize, blockSize>>>(d_weights, d_outputs[layer], d_errors[layer], layer);
            hipDeviceSynchronize();
        }
    }

    for (int i = 0; i < NUM_LAYERS; i++) {
        if (i > 0) hipFree(d_outputs[i]);
        if (i < NUM_LAYERS - 1) hipFree(d_errors[i]);
    }
    hipFree(d_outputs[0]);
    hipFree(d_errors[NUM_LAYERS - 2]);
}

int main() {
    float *h_input = new float[NUM_NEURONS * BATCH_SIZE];
    float *h_target = new float[NUM_NEURONS * BATCH_SIZE];
    float *h_weights = new float[NUM_LAYERS * NUM_NEURONS * NUM_NEURONS];

    // Initialize input and target data
    for (int i = 0; i < NUM_NEURONS * BATCH_SIZE; i++) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
        h_target[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float *d_input, *d_target, *d_weights;
    hipMalloc(&d_input, NUM_NEURONS * BATCH_SIZE * sizeof(float));
    hipMalloc(&d_target, NUM_NEURONS * BATCH_SIZE * sizeof(float));
    hipMalloc(&d_weights, NUM_LAYERS * NUM_NEURONS * NUM_NEURONS * sizeof(float));

    hipMemcpy(d_input, h_input, NUM_NEURONS * BATCH_SIZE * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_target, h_target, NUM_NEURONS * BATCH_SIZE * sizeof(float), hipMemcpyHostToDevice);

    // Initialize weights
    int num_weights = NUM_LAYERS * NUM_NEURONS * NUM_NEURONS;
    initialize_weights<<<(num_weights + 255) / 256, 256>>>(d_weights, num_weights);
    hipDeviceSynchronize();

    // Train the neural network
    train_neural_network(d_input, d_target, d_weights, 100);

    hipMemcpy(h_weights, d_weights, NUM_LAYERS * NUM_NEURONS * NUM_NEURONS * sizeof(float), hipMemcpyDeviceToHost);

    // Clean up
    delete[] h_input;
    delete[] h_target;
    delete[] h_weights;
    hipFree(d_input);
    hipFree(d_target);
    hipFree(d_weights);

    return 0;
}