#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define NUM_PATHS 128
#define NUM_STEPS 1024
#define BLOCK_SIZE 256

// Define a simple option pricing structure
struct OptionPricingData {
    float S0; // Initial stock price
    float r;  // Risk-free rate
    float sigma; // Volatility
    float T;  // Time to maturity
    float K;  // Strike price
};

// Kernel to simulate stock prices and compute option payoff
__global__ void monteCarloKernel(hiprandState *states, OptionPricingData data, float *payoffs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_PATHS) return;

    hiprandState localState = states[idx];
    float S = data.S0;
    float dt = data.T / NUM_STEPS;

    // Simulate stock price path
    for (int step = 0; step < NUM_STEPS; ++step) {
        float epsilon = hiprand_normal(&localState);
        S *= exp((data.r - 0.5f * data.sigma * data.sigma) * dt + data.sigma * sqrtf(dt) * epsilon);
    }

    // Calculate payoff for this path
    float payoff = fmaxf(0.0f, S - data.K);
    payoffs[idx] = payoff;

    // Write back the state
    states[idx] = localState;
}

// Kernel to initialize random states
__global__ void initCurandStates(hiprandState *states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_PATHS) return;
    hiprand_init(seed, idx, 0, &states[idx]);
}

int main() {
    // Allocate memory for random states on device
    hiprandState *d_states;
    hipMalloc((void **)&d_states, NUM_PATHS * sizeof(hiprandState));
    if (d_states == nullptr) {
        std::cerr << "Failed to allocate memory for random states" << std::endl;
        return -1;
    }

    // Allocate memory for payoffs on device
    float *d_payoffs;
    hipMalloc((void **)&d_payoffs, NUM_PATHS * sizeof(float));
    if (d_payoffs == nullptr) {
        std::cerr << "Failed to allocate memory for payoffs" << std::endl;
        hipFree(d_states);
        return -1;
    }

    // Initialize random states
    initCurandStates<<<(NUM_PATHS + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_states, 1234);
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel launch failed: " << hipGetErrorString(err) << std::endl;
        hipFree(d_states);
        hipFree(d_payoffs);
        return -1;
    }

    // Set up option pricing data
    OptionPricingData data = {100.0f, 0.05f, 0.2f, 1.0f, 100.0f};
    OptionPricingData *d_data;
    hipMalloc((void **)&d_data, sizeof(OptionPricingData));
    hipMemcpy(d_data, &data, sizeof(OptionPricingData), hipMemcpyHostToDevice);

    // Run Monte Carlo simulation
    monteCarloKernel<<<(NUM_PATHS + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_states, *d_data, d_payoffs);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel launch failed: " << hipGetErrorString(err) << std::endl;
        hipFree(d_states);
        hipFree(d_payoffs);
        hipFree(d_data);
        return -1;
    }

    // Copy results back to host
    float *h_payoffs = new float[NUM_PATHS];
    hipMemcpy(h_payoffs, d_payoffs, NUM_PATHS * sizeof(float), hipMemcpyDeviceToHost);

    // Calculate average payoff
    float sum = 0.0f;
    for (int i = 0; i < NUM_PATHS; ++i) {
        sum += h_payoffs[i];
    }
    float averagePayoff = sum / NUM_PATHS;
    float optionPrice = expf(-data.r * data.T) * averagePayoff;

    // Output the result
    std::cout << "Estimated option price: " << optionPrice << std::endl;

    // Clean up
    delete[] h_payoffs;
    hipFree(d_states);
    hipFree(d_payoffs);
    hipFree(d_data);

    return 0;
}