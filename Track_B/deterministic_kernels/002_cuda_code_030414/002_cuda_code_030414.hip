#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define PATHS_PER_BLOCK 32
#define BLOCKS 16
#define PATHS (PATHS_PER_BLOCK * BLOCKS)
#define STEPS 1024

__global__ void monte_carlo_kernel(hiprandState *states, float *prices, float S0, float mu, float sigma, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= PATHS) return;

    hiprandState localState = states[idx];
    float S = S0;

    for (int step = 0; step < STEPS; step++) {
        float epsilon = hiprand_normal(&localState);
        S *= exp((mu - 0.5f * sigma * sigma) * dt + sigma * sqrtf(dt) * epsilon);
    }

    prices[idx] = S;
    states[idx] = localState;
}

__global__ void initialize_states(hiprandState *states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= PATHS) return;

    hiprand_init(seed, idx, 0, &states[idx]);
}

int main() {
    float *prices;
    hiprandState *states;

    // Allocate Unified Memory – accessible from CPU or GPU
    hipError_t err = hipMallocManaged(&prices, PATHS * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate prices: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    err = hipMallocManaged(&states, PATHS * sizeof(hiprandState));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate states: " << hipGetErrorString(err) << std::endl;
        hipFree(prices);
        return -1;
    }

    // Initialize states
    initialize_states<<<BLOCKS, PATHS_PER_BLOCK>>>(states, 1234);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch initialize_states kernel: " << hipGetErrorString(err) << std::endl;
        hipFree(prices);
        hipFree(states);
        return -1;
    }

    // Launch Monte Carlo kernel
    float S0 = 100.0f;  // Initial stock price
    float mu = 0.05f;   // Drift coefficient
    float sigma = 0.2f; // Volatility
    float dt = 0.01f;   // Time step

    monte_carlo_kernel<<<BLOCKS, PATHS_PER_BLOCK>>>(states, prices, S0, mu, sigma, dt);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch monte_carlo_kernel kernel: " << hipGetErrorString(err) << std::endl;
        hipFree(prices);
        hipFree(states);
        return -1;
    }

    // Wait for GPU to finish before accessing on host
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        std::cerr << "Failed to synchronize: " << hipGetErrorString(err) << std::endl;
        hipFree(prices);
        hipFree(states);
        return -1;
    }

    // Output the results
    std::cout << "Monte Carlo Simulation Results:\n";
    for (int i = 0; i < PATHS; i++) {
        std::cout << "Path " << i << ": " << prices[i] << "\n";
    }

    // Free memory
    hipFree(prices);
    hipFree(states);

    return 0;
}