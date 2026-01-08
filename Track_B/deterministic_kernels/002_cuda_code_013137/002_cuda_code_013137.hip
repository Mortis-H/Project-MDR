#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>
#include <math.h>

#define NUM_STATES 8192
#define BLOCK_SIZE 256

__global__ void initializeStates(hiprandState *states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_STATES) {
        hiprand_init(seed, idx, 0, &states[idx]);
    }
}

__device__ float energyFunction(int state) {
    // Example energy function: simple quadratic function
    return static_cast<float>(state * state);
}

__device__ float acceptanceProbability(float oldEnergy, float newEnergy, float temperature) {
    if (newEnergy < oldEnergy) {
        return 1.0f;
    } else {
        return expf((oldEnergy - newEnergy) / temperature);
    }
}

__global__ void simulatedAnnealingKernel(hiprandState *states, int *statesArray, float initialTemperature, float coolingRate) {
    extern __shared__ int sharedStates[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_STATES) return;

    // Load state into shared memory for coalesced access
    sharedStates[threadIdx.x] = statesArray[idx];
    __syncthreads();

    int currentState = sharedStates[threadIdx.x];
    float currentEnergy = energyFunction(currentState);
    float temperature = initialTemperature;

    while (temperature > 1e-5f) {
        // Generate a new state
        int newState = hiprand(&states[idx]) % NUM_STATES;
        float newEnergy = energyFunction(newState);

        // Acceptance criteria
        float ap = acceptanceProbability(currentEnergy, newEnergy, temperature);
        float r = hiprand_uniform(&states[idx]);
        if (r < ap) {
            currentState = newState;
            currentEnergy = newEnergy;
        }

        temperature *= coolingRate;
    }

    // Write back the result
    statesArray[idx] = currentState;
}

int main() {
    int *h_states = new int[NUM_STATES];
    int *d_states;
    hiprandState *d_states_random;

    // Initialize host states
    for (int i = 0; i < NUM_STATES; i++) {
        h_states[i] = i % 256; // Simple initialization
    }

    // Allocate device memory
    hipError_t err = hipMalloc(&d_states, NUM_STATES * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Error allocating memory for states: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc(&d_states_random, NUM_STATES * sizeof(hiprandState));
    if (err != hipSuccess) {
        std::cerr << "Error allocating memory for random states: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy states to device
    err = hipMemcpy(d_states, h_states, NUM_STATES * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Error copying states to device: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Initialize random states
    initializeStates<<<(NUM_STATES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_states_random, 1234);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error initializing random states: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Run simulated annealing kernel
    simulatedAnnealingKernel<<<(NUM_STATES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(d_states_random, d_states, 1000.0f, 0.995f);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error in simulated annealing kernel: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy results back to host
    err = hipMemcpy(h_states, d_states, NUM_STATES * sizeof(int), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Error copying states back to host: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Output first 10 results for verification
    std::cout << "First 10 states after annealing:" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << h_states[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    hipFree(d_states);
    hipFree(d_states_random);

    // Free host memory
    delete[] h_states;

    return 0;
}