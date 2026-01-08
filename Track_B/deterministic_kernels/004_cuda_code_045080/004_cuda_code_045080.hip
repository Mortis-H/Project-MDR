#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define NUM_LAYERS 5
#define NUM_NEURONS 64
#define BATCH_SIZE 256
#define EPOCHS 100
#define LEARNING_RATE 0.01f

__global__ void initialize_weights(float* weights, int num_weights) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_weights) {
        hiprandState state;
        hiprand_init((unsigned long long)clock() + idx, 0, 0, &state);
        weights[idx] = hiprand_uniform(&state) * 0.01f;
    }
}

__global__ void forward_pass(float* inputs, float* weights, float* activations, int num_layers, int num_neurons, int batch_size) {
    extern __shared__ float shared_weights[];
    int neuron_idx = threadIdx.x;
    int batch_idx = blockIdx.x;

    if (neuron_idx < num_neurons) {
        for (int layer = 0; layer < num_layers; ++layer) {
            float sum = 0.0f;
            int base_weight_idx = layer * num_neurons * num_neurons;
            for (int prev_neuron = 0; prev_neuron < num_neurons; ++prev_neuron) {
                int weight_idx = base_weight_idx + neuron_idx * num_neurons + prev_neuron;
                shared_weights[neuron_idx * num_neurons + prev_neuron] = weights[weight_idx];
            }
            __syncthreads();

            for (int prev_neuron = 0; prev_neuron < num_neurons; ++prev_neuron) {
                sum += shared_weights[neuron_idx * num_neurons + prev_neuron] * activations[batch_idx * num_neurons + prev_neuron];
            }
            activations[batch_idx * num_neurons + neuron_idx] = 1.0f / (1.0f + expf(-sum)); // Sigmoid activation
        }
    }
}

__global__ void backward_pass(float* inputs, float* weights, float* activations, float* gradients, int num_layers, int num_neurons, int batch_size) {
    extern __shared__ float shared_weights[];
    int neuron_idx = threadIdx.x;
    int batch_idx = blockIdx.x;

    if (neuron_idx < num_neurons) {
        for (int layer = num_layers - 1; layer >= 0; --layer) {
            float error = 0.0f;
            int base_weight_idx = layer * num_neurons * num_neurons;

            if (layer == num_layers - 1) {
                error = activations[batch_idx * num_neurons + neuron_idx] - inputs[batch_idx * num_neurons + neuron_idx];
            } else {
                for (int next_neuron = 0; next_neuron < num_neurons; ++next_neuron) {
                    int weight_idx = base_weight_idx + next_neuron * num_neurons + neuron_idx;
                    shared_weights[neuron_idx * num_neurons + next_neuron] = weights[weight_idx];
                }
                __syncthreads();

                for (int next_neuron = 0; next_neuron < num_neurons; ++next_neuron) {
                    error += shared_weights[neuron_idx * num_neurons + next_neuron] * gradients[batch_idx * num_neurons + next_neuron];
                }
            }

            float output = activations[batch_idx * num_neurons + neuron_idx];
            float delta = error * output * (1.0f - output);

            for (int prev_neuron = 0; prev_neuron < num_neurons; ++prev_neuron) {
                int weight_idx = base_weight_idx + neuron_idx * num_neurons + prev_neuron;
                atomicAdd(&gradients[batch_idx * num_neurons + prev_neuron], delta * activations[batch_idx * num_neurons + prev_neuron]);
            }
            gradients[batch_idx * num_neurons + neuron_idx] = delta;
        }
    }
}

__global__ void update_weights(float* weights, float* gradients, int num_weights, float learning_rate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_weights) {
        atomicAdd(&weights[idx], -learning_rate * gradients[idx]);
    }
}

int main() {
    float* d_inputs;
    float* d_weights;
    float* d_activations;
    float* d_gradients;

    int num_weights = NUM_LAYERS * NUM_NEURONS * NUM_NEURONS;
    size_t size_weights = num_weights * sizeof(float);
    size_t size_inputs = BATCH_SIZE * NUM_NEURONS * sizeof(float);
    size_t size_activations = BATCH_SIZE * NUM_NEURONS * sizeof(float);
    size_t size_gradients = BATCH_SIZE * NUM_NEURONS * sizeof(float);

    hipError_t err = hipMalloc(&d_inputs, size_inputs);
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate memory for inputs: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    err = hipMalloc(&d_weights, size_weights);
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate memory for weights: " << hipGetErrorString(err) << std::endl;
        hipFree(d_inputs);
        return -1;
    }

    err = hipMalloc(&d_activations, size_activations);
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate memory for activations: " << hipGetErrorString(err) << std::endl;
        hipFree(d_inputs);
        hipFree(d_weights);
        return -1;
    }

    err = hipMalloc(&d_gradients, size_gradients);
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate memory for gradients: " << hipGetErrorString(err) << std::endl;
        hipFree(d_inputs);
        hipFree(d_weights);
        hipFree(d_activations);
        return -1;
    }

    int threads_per_block = NUM_NEURONS;
    int blocks_per_grid = BATCH_SIZE;

    initialize_weights<<<blocks_per_grid, threads_per_block>>>(d_inputs, BATCH_SIZE * NUM_NEURONS);
    hipDeviceSynchronize();

    initialize_weights<<<(num_weights + threads_per_block - 1) / threads_per_block, threads_per_block>>>(d_weights, num_weights);
    hipDeviceSynchronize();

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        forward_pass<<<blocks_per_grid, threads_per_block, threads_per_block * threads_per_block * sizeof(float)>>>(d_inputs, d_weights, d_activations, NUM_LAYERS, NUM_NEURONS, BATCH_SIZE);
        hipDeviceSynchronize();

        backward_pass<<<blocks_per_grid, threads_per_block, threads_per_block * threads_per_block * sizeof(float)>>>(d_inputs, d_weights, d_activations, d_gradients, NUM_LAYERS, NUM_NEURONS, BATCH_SIZE);
        hipDeviceSynchronize();

        update_weights<<<(num_weights + threads_per_block - 1) / threads_per_block, threads_per_block>>>(d_weights, d_gradients, num_weights, LEARNING_RATE);
        hipDeviceSynchronize();
    }

    hipFree(d_inputs);
    hipFree(d_weights);
    hipFree(d_activations);
    hipFree(d_gradients);

    return 0;
}