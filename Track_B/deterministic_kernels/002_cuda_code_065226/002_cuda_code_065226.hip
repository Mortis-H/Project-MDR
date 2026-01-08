#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

// Define the number of paths and steps
#define NUM_PATHS 512
#define NUM_STEPS 1024

// Kernel to initialize the random number generator states
__global__ void initCurandStates(hiprandState *states, unsigned long seed) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < NUM_PATHS) {
        hiprand_init(seed, id, 0, &states[id]);
    }
}

// Kernel to perform Monte Carlo simulation for option pricing
__global__ void monteCarloSimulation(hiprandState *states, float *d_results, float S0, float r, float sigma, float T) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < NUM_PATHS) {
        float S = S0;
        float dt = T / NUM_STEPS;
        for (int step = 0; step < NUM_STEPS; step++) {
            float epsilon = hiprand_normal(&states[id]);
            S *= exp((r - 0.5f * sigma * sigma) * dt + sigma * sqrtf(dt) * epsilon);
        }
        d_results[id] = fmaxf(0.0f, S - 100.0f); // Assuming strike price is 100
    }
}

int main() {
    // Device pointers
    hiprandState *d_states;
    float *d_results;

    // Host pointers
    float *h_results = new float[NUM_PATHS];

    // Allocate memory on device
    hipError_t err = hipMalloc(&d_states, NUM_PATHS * sizeof(hiprandState));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate memory for random states" << std::endl;
        return -1;
    }

    err = hipMalloc(&d_results, NUM_PATHS * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate memory for results" << std::endl;
        hipFree(d_states);
        return -1;
    }

    // Initialize random number generator states
    unsigned long seed = 1234;
    initCurandStates<<<(NUM_PATHS + 255) / 256, 256>>>(d_states, seed);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to initialize random states" << std::endl;
        hipFree(d_states);
        hipFree(d_results);
        return -1;
    }

    // Define parameters for the option pricing
    float S0 = 100.0f; // Initial stock price
    float r = 0.05f;   // Risk-free rate
    float sigma = 0.2f; // Volatility
    float T = 1.0f;    // Time to maturity

    // Perform Monte Carlo simulation
    monteCarloSimulation<<<(NUM_PATHS + 255) / 256, 256>>>(d_states, d_results, S0, r, sigma, T);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to execute Monte Carlo simulation" << std::endl;
        hipFree(d_states);
        hipFree(d_results);
        return -1;
    }

    // Asynchronous memory copy to overlap computation and data transfer
    hipMemcpyAsync(h_results, d_results, NUM_PATHS * sizeof(float), hipMemcpyDeviceToHost, hipStreamDefault);

    // Synchronize to ensure all operations are complete
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        std::cerr << "Failed to synchronize device" << std::endl;
        hipFree(d_states);
        hipFree(d_results);
        return -1;
    }

    // Calculate the average payoff
    float sum = 0.0f;
    for (int i = 0; i < NUM_PATHS; i++) {
        sum += h_results[i];
    }
    float optionPrice = expf(-r * T) * sum / NUM_PATHS;

    // Output the result
    std::cout << "Estimated Option Price: " << optionPrice << std::endl;

    // Free allocated memory
    delete[] h_results;
    hipFree(d_states);
    hipFree(d_results);

    return 0;
}