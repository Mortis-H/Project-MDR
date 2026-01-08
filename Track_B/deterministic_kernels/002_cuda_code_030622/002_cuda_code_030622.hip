
#include <hip/hip_runtime.h>
#include <iostream>
#include <hiprand/hiprand_kernel.h>

#define PATHS_PER_BLOCK 256
#define PATHS 2048

// Kernel to perform Monte Carlo simulation for option pricing
__global__ void monteCarloSimulation(hiprandState *states, float *prices, float S0, float mu, float sigma, float T, int steps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= PATHS) return;

    float S = S0;
    float dt = T / steps;
    float sqrt_dt = sqrt(dt);

    for (int i = 0; i < steps; ++i) {
        float Z = hiprand_normal(&states[idx]);
        S *= exp((mu - 0.5f * sigma * sigma) * dt + sigma * Z * sqrt_dt);
    }

    prices[idx] = S;
}

// Function to initialize random states
__global__ void initRandomStates(hiprandState *states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= PATHS) return;
    hiprand_init(seed, idx, 0, &states[idx]);
}

int main() {
    const int paths = PATHS;
    const int blockSize = PATHS_PER_BLOCK;
    const int numBlocks = (paths + blockSize - 1) / blockSize;

    float S0 = 100.0f; // Initial stock price
    float mu = 0.05f;  // Drift coefficient
    float sigma = 0.2f; // Volatility
    float T = 1.0f;    // Time to maturity
    int steps = 252;   // Number of time steps

    float *prices;
    hiprandState *states;

    // Allocate memory for prices and random states
    hipError_t err = hipMalloc(&prices, paths * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Error allocating memory for prices: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc(&states, paths * sizeof(hiprandState));
    if (err != hipSuccess) {
        std::cerr << "Error allocating memory for states: " << hipGetErrorString(err) << std::endl;
        hipFree(prices);
        return -1;
    }

    // Initialize random states
    initRandomStates<<<numBlocks, blockSize>>>(states, 1234);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error initializing random states: " << hipGetErrorString(err) << std::endl;
        hipFree(prices);
        hipFree(states);
        return -1;
    }

    // Perform Monte Carlo simulation
    monteCarloSimulation<<<numBlocks, blockSize>>>(states, prices, S0, mu, sigma, T, steps);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error in Monte Carlo simulation: " << hipGetErrorString(err) << std::endl;
        hipFree(prices);
        hipFree(states);
        return -1;
    }

    // Copy results back to host
    float *h_prices = new float[paths];
    err = hipMemcpy(h_prices, prices, paths * sizeof(float), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Error copying results from device to host: " << hipGetErrorString(err) << std::endl;
        delete[] h_prices;
        hipFree(prices);
        hipFree(states);
        return -1;
    }

    // Print first 10 prices for demonstration
    std::cout << "First 10 simulated stock prices:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << h_prices[i] << std::endl;
    }

    // Free allocated memory
    delete[] h_prices;
    hipFree(prices);
    hipFree(states);

    return 0;
}