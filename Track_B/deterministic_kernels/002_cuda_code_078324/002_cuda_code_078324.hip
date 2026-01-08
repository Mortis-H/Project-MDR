#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>
#include <cmath>

#define NUM_STATES 4096
#define BLOCK_SIZE 256

// Kernel to initialize random states
__global__ void initStates(hiprandState *states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_STATES) {
        hiprand_init(seed, idx, 0, &states[idx]);
    }
}

// Simulated annealing kernel
__global__ void simulatedAnnealing(hiprandState *states, float *energies, float temperature) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_STATES) {
        // Simulate a state change
        float currentEnergy = energies[idx];
        float proposedEnergy = currentEnergy + (hiprand_normal(&states[idx]) * 0.1f);

        // Metropolis criterion
        if (proposedEnergy < currentEnergy || exp((currentEnergy - proposedEnergy) / temperature) > hiprand_uniform(&states[idx])) {
            energies[idx] = proposedEnergy;
        }
    }
}

int main() {
    // Allocate memory on host
    float *h_energies = new float[NUM_STATES];
    for (int i = 0; i < NUM_STATES; ++i) {
        h_energies[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate memory on device
    float *d_energies;
    hiprandState *d_states;
    hipMalloc(&d_energies, NUM_STATES * sizeof(float));
    hipMalloc(&d_states, NUM_STATES * sizeof(hiprandState));

    // Copy initial energies to device
    hipMemcpy(d_energies, h_energies, NUM_STATES * sizeof(float), hipMemcpyHostToDevice);

    // Initialize random states on device
    initStates<<<(NUM_STATES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_states, time(NULL));
    hipDeviceSynchronize();

    // Perform simulated annealing
    for (float temperature = 1.0f; temperature > 0.01f; temperature *= 0.99f) {
        // Asynchronous memory copy to overlap with computation
        hipMemcpyAsync(d_energies, h_energies, NUM_STATES * sizeof(float), hipMemcpyHostToDevice, 0);

        // Launch kernel
        simulatedAnnealing<<<(NUM_STATES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_states, d_energies, temperature);
        hipDeviceSynchronize();

        // Asynchronous memory copy to overlap with computation
        hipMemcpyAsync(h_energies, d_energies, NUM_STATES * sizeof(float), hipMemcpyDeviceToHost, 0);
    }

    // Free device memory
    hipFree(d_energies);
    hipFree(d_states);

    // Free host memory
    delete[] h_energies;

    // Check for any errors launching the kernel
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch kernel (error code " << err << "): " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    std::cout << "Simulated Annealing completed successfully." << std::endl;
    return 0;
}