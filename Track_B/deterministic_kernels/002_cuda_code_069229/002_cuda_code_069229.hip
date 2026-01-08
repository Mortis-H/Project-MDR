#include <iostream>
#include <hiprand/hiprand_kernel.h>
#include <hip/hip_runtime.h>

#define PATHS_PER_BLOCK 8
#define BLOCKS 8
#define PATHS (PATHS_PER_BLOCK * BLOCKS)
#define STEPS 1024
#define MAX_LEVEL 4

__global__ void monteCarloSimulation(hiprandState *states, float *prices, float S0, float r, float sigma, float T, int steps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= PATHS) return;

    hiprandState localState = states[idx];
    float S = S0;
    float dt = T / steps;
    float sqrt_dt = sqrt(dt);

    for (int i = 0; i < steps; ++i) {
        float Z = hiprand_normal(&localState);
        S *= exp((r - 0.5f * sigma * sigma) * dt + sigma * sqrt_dt * Z);
    }

    prices[idx] = S;
    states[idx] = localState;
}

__global__ void initializeStates(hiprandState *states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= PATHS) return;
    hiprand_init(seed, idx, 0, &states[idx]);
}

int main() {
    const float S0 = 100.0f; // Initial stock price
    const float r = 0.05f;   // Risk-free rate
    const float sigma = 0.2f; // Volatility
    const float T = 1.0f;    // Time to maturity
    const int steps = STEPS; // Number of steps

    float *d_prices;
    hiprandState *d_states;

    // Allocate memory on the device
    hipError_t err = hipMalloc((void **)&d_prices, PATHS * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Error allocating memory for prices: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    err = hipMalloc((void **)&d_states, PATHS * sizeof(hiprandState));
    if (err != hipSuccess) {
        std::cerr << "Error allocating memory for states: " << hipGetErrorString(err) << std::endl;
        hipFree(d_prices);
        return -1;
    }

    // Initialize random states
    initializeStates<<<BLOCKS, PATHS_PER_BLOCK>>>(d_states, 12345);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error initializing random states: " << hipGetErrorString(err) << std::endl;
        hipFree(d_prices);
        hipFree(d_states);
        return -1;
    }

    // Launch the kernel
    monteCarloSimulation<<<BLOCKS, PATHS_PER_BLOCK>>>(d_states, d_prices, S0, r, sigma, T, steps);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error launching kernel: " << hipGetErrorString(err) << std::endl;
        hipFree(d_prices);
        hipFree(d_states);
        return -1;
    }

    // Copy results back to host
    float *h_prices = new float[PATHS];
    err = hipMemcpy(h_prices, d_prices, PATHS * sizeof(float), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Error copying results back to host: " << hipGetErrorString(err) << std::endl;
        delete[] h_prices;
        hipFree(d_prices);
        hipFree(d_states);
        return -1;
    }

    // Print the results
    for (int i = 0; i < PATHS; ++i) {
        std::cout << "Path " << i << ": " << h_prices[i] << std::endl;
    }

    // Free device memory
    hipFree(d_prices);
    hipFree(d_states);

    // Free host memory
    delete[] h_prices;

    return 0;
}