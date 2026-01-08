#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

// Define constants
#define PATHS_PER_BLOCK 256
#define BLOCKS_PER_GPU 64
#define TOTAL_PATHS (PATHS_PER_BLOCK * BLOCKS_PER_GPU)
#define NUM_GPUS 2

// Kernel to perform Monte Carlo simulation for option pricing
__global__ void monteCarloSimulation(hiprandState *states, float *prices, float S0, float sigma, float r, float T, int steps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= TOTAL_PATHS) return;

    float S = S0;
    float dt = T / steps;
    float sqrt_dt = sqrt(dt);

    // Generate random numbers and simulate stock price path
    for (int i = 0; i < steps; ++i) {
        float x = hiprand_normal(&states[idx]);
        S *= exp((r - 0.5f * sigma * sigma) * dt + sigma * x * sqrt_dt);
    }

    // Calculate payoff for a call option with strike price K
    float K = 100.0f;
    prices[idx] = fmaxf(0.0f, S - K);
}

// Initialize random number states
__global__ void initCurandStates(hiprandState *states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= TOTAL_PATHS) return;
    hiprand_init(seed, idx, 0, &states[idx]);
}

int main() {
    // Device properties
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    std::cout << "Using device: " << prop.name << std::endl;

    // Parameters for option pricing
    float S0 = 100.0f;  // Initial stock price
    float sigma = 0.2f; // Volatility
    float r = 0.05f;    // Risk-free rate
    float T = 1.0f;     // Time to maturity
    int steps = 100;    // Number of steps in the simulation

    // Allocate memory for random states and prices
    hiprandState *d_states;
    float *d_prices;
    hipMalloc(&d_states, TOTAL_PATHS * sizeof(hiprandState));
    hipMalloc(&d_prices, TOTAL_PATHS * sizeof(float));

    // Initialize random states
    initCurandStates<<<BLOCKS_PER_GPU, PATHS_PER_BLOCK>>>(d_states, 1234);
    hipDeviceSynchronize();
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Perform Monte Carlo simulation
    monteCarloSimulation<<<BLOCKS_PER_GPU, PATHS_PER_BLOCK>>>(d_states, d_prices, S0, sigma, r, T, steps);
    hipDeviceSynchronize();
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy results back to host
    float *h_prices = new float[TOTAL_PATHS];
    hipMemcpy(h_prices, d_prices, TOTAL_PATHS * sizeof(float), hipMemcpyDeviceToHost);

    // Calculate the average payoff
    float sum = 0.0f;
    for (int i = 0; i < TOTAL_PATHS; ++i) {
        sum += h_prices[i];
    }
    float option_price = exp(-r * T) * sum / TOTAL_PATHS;
    std::cout << "Estimated option price: " << option_price << std::endl;

    // Free memory
    delete[] h_prices;
    hipFree(d_states);
    hipFree(d_prices);

    return 0;
}