#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

// Define constants
#define NUM_PATHS 1024
#define NUM_STEPS 256
#define BLOCK_SIZE 256

// Kernel to initialize random states
__global__ void initRandomStates(hiprandState *state, unsigned long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < NUM_PATHS) {
        hiprand_init(seed, idx, 0, &state[idx]);
    }
}

// Kernel to perform Monte Carlo simulation for option pricing
__global__ void monteCarloSimulation(hiprandState *state, float *prices, float S0, float r, float sigma, float T) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < NUM_PATHS) {
        float S = S0;
        float dt = T / NUM_STEPS;
        float sqrt_dt = sqrt(dt);
        for (int step = 0; step < NUM_STEPS; ++step) {
            float Z = hiprand_normal(&state[idx]);
            S *= exp((r - 0.5f * sigma * sigma) * dt + sigma * sqrt_dt * Z);
        }
        prices[idx] = S;
    }
}

// Function to check CUDA errors
void checkCudaError(hipError_t error, const char *message) {
    if (error != hipSuccess) {
        std::cerr << "CUDA error: " << message << " - " << hipGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Allocate memory for random states and prices on host
    hiprandState *d_state;
    float *d_prices, *h_prices;
    h_prices = (float *)malloc(NUM_PATHS * sizeof(float));

    // Allocate memory for random states and prices on device
    checkCudaError(hipMalloc(&d_state, NUM_PATHS * sizeof(hiprandState)), "Failed to allocate memory for random states");
    checkCudaError(hipMalloc(&d_prices, NUM_PATHS * sizeof(float)), "Failed to allocate memory for prices");

    // Initialize random states
    initRandomStates<<<(NUM_PATHS + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_state, 1234);
    checkCudaError(hipGetLastError(), "Kernel launch failed for initRandomStates");

    // Start asynchronous memory copy
    hipStream_t stream;
    checkCudaError(hipStreamCreate(&stream), "Failed to create stream");
    checkCudaError(hipMemcpyAsync(d_prices, h_prices, NUM_PATHS * sizeof(float), hipMemcpyHostToDevice, stream), "Failed to copy prices to device");

    // Perform Monte Carlo simulation
    monteCarloSimulation<<<(NUM_PATHS + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(d_state, d_prices, 100.0f, 0.05f, 0.2f, 1.0f);
    checkCudaError(hipGetLastError(), "Kernel launch failed for monteCarloSimulation");

    // Copy results back to host
    checkCudaError(hipMemcpyAsync(h_prices, d_prices, NUM_PATHS * sizeof(float), hipMemcpyDeviceToHost, stream), "Failed to copy prices back to host");

    // Synchronize stream
    checkCudaError(hipStreamSynchronize(stream), "Failed to synchronize stream");

    // Clean up
    checkCudaError(hipFree(d_state), "Failed to free device memory for random states");
    checkCudaError(hipFree(d_prices), "Failed to free device memory for prices");
    checkCudaError(hipStreamDestroy(stream), "Failed to destroy stream");
    free(h_prices);

    // Output the first few results for verification
    std::cout << "First 10 simulated prices:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << h_prices[i] << std::endl;
    }

    return 0;
}