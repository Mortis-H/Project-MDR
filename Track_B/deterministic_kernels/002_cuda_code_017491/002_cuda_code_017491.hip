#include <iostream>
#include <hiprand/hiprand_kernel.h>
#include <hip/hip_runtime.h>

#define NUM_STATES 512
#define BLOCK_SIZE 256

// Kernel to initialize random states using cuRAND
__global__ void initRandomStates(hiprandState *states, unsigned long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < NUM_STATES) {
        hiprand_init(seed, idx, 0, &states[idx]);
    }
}

// Simulated annealing kernel
__global__ void simulatedAnnealing(hiprandState *states, float *energies, float *temperatures, float coolingRate) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= NUM_STATES) return;

    // Load initial energy and temperature
    float currentEnergy = energies[idx];
    float temperature = temperatures[idx];

    // Perform simulated annealing
    while (temperature > 1e-3) {
        // Generate a random neighbor and its energy
        float newEnergy = currentEnergy + (hiprand_normal(&states[idx]) * 0.1f);

        // Calculate the energy difference
        float deltaEnergy = newEnergy - currentEnergy;

        // Metropolis criterion
        if (deltaEnergy < 0 || expf(-deltaEnergy / temperature) > hiprand_uniform(&states[idx])) {
            currentEnergy = newEnergy;
        }

        // Cool down
        temperature *= coolingRate;
    }

    // Store the final energy
    energies[idx] = currentEnergy;
}

int main() {
    // Allocate memory for random states and energies
    hiprandState *d_states;
    float *d_energies, *d_temperatures;
    float *h_energies = new float[NUM_STATES];
    float *h_temperatures = new float[NUM_STATES];
    float coolingRate = 0.99f;

    // Initialize energies and temperatures on host
    for (int i = 0; i < NUM_STATES; ++i) {
        h_energies[i] = 100.0f; // Initial high energy
        h_temperatures[i] = 100.0f; // Initial high temperature
    }

    // Allocate device memory
    hipError_t err = hipMalloc((void **)&d_states, NUM_STATES * sizeof(hiprandState));
    if (err != hipSuccess) {
        std::cerr << "CUDA error allocating memory for states: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc((void **)&d_energies, NUM_STATES * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "CUDA error allocating memory for energies: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc((void **)&d_temperatures, NUM_STATES * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "CUDA error allocating memory for temperatures: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy initial energies and temperatures to device
    err = hipMemcpy(d_energies, h_energies, NUM_STATES * sizeof(float), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "CUDA error copying energies to device: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMemcpy(d_temperatures, h_temperatures, NUM_STATES * sizeof(float), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "CUDA error copying temperatures to device: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Initialize random states
    initRandomStates<<<(NUM_STATES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_states, 1234);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error initializing random states: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Run simulated annealing
    simulatedAnnealing<<<(NUM_STATES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_states, d_energies, d_temperatures, coolingRate);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error during simulated annealing: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy final energies back to host
    err = hipMemcpy(h_energies, d_energies, NUM_STATES * sizeof(float), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "CUDA error copying energies back to host: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Free device memory
    hipFree(d_states);
    hipFree(d_energies);
    hipFree(d_temperatures);

    // Output final energies (first 10 for brevity)
    std::cout << "Final energies of first 10 states:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << "State " << i << ": " << h_energies[i] << std::endl;
    }

    // Clean up
    delete[] h_energies;
    delete[] h_temperatures;

    return 0;
}