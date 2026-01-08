// genetic_algorithm.cu
#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

// Define constants
#define POPULATION_SIZE 8192
#define GENES_PER_INDIVIDUAL 32
#define BLOCK_SIZE 256

// Kernel to perform genetic algorithm operations
__global__ void geneticAlgorithmKernel(float* population, float* fitness, hiprandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < POPULATION_SIZE) {
        // Initialize random number generator
        hiprandState localState = states[idx];
        float mutationRate = 0.01f;

        // Simulate genetic operations: crossover and mutation
        for (int gene = 0; gene < GENES_PER_INDIVIDUAL; ++gene) {
            // Perform crossover with a random individual
            int mateIdx = hiprand(&localState) % POPULATION_SIZE;
            float mateGene = population[mateIdx * GENES_PER_INDIVIDUAL + gene];
            float myGene = population[idx * GENES_PER_INDIVIDUAL + gene];
            population[idx * GENES_PER_INDIVIDUAL + gene] = (hiprand(&localState) < 0.5f) ? myGene : mateGene;

            // Perform mutation
            if (hiprand_uniform(&localState) < mutationRate) {
                population[idx * GENES_PER_INDIVIDUAL + gene] = hiprand_uniform(&localState);
            }
        }

        // Update fitness (dummy fitness calculation)
        float total = 0.0f;
        for (int gene = 0; gene < GENES_PER_INDIVIDUAL; ++gene) {
            total += population[idx * GENES_PER_INDIVIDUAL + gene];
        }
        fitness[idx] = total / GENES_PER_INDIVIDUAL;

        // Write back the state
        states[idx] = localState;
    }
}

// Initialize random states for each thread
__global__ void initCurandStates(hiprandState* states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        hiprand_init(seed, idx, 0, &states[idx]);
    }
}

int main() {
    // Allocate memory on host
    float* h_population = new float[POPULATION_SIZE * GENES_PER_INDIVIDUAL];
    float* h_fitness = new float[POPULATION_SIZE];
    hiprandState* h_states = new hiprandState[POPULATION_SIZE];

    // Initialize population with random values
    for (int i = 0; i < POPULATION_SIZE * GENES_PER_INDIVIDUAL; ++i) {
        h_population[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate memory on device
    float* d_population;
    float* d_fitness;
    hiprandState* d_states;
    hipError_t err;

    err = hipMalloc((void**)&d_population, POPULATION_SIZE * GENES_PER_INDIVIDUAL * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate d_population (error code " << err << ")" << std::endl;
        return -1;
    }

    err = hipMalloc((void**)&d_fitness, POPULATION_SIZE * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate d_fitness (error code " << err << ")" << std::endl;
        return -1;
    }

    err = hipMalloc((void**)&d_states, POPULATION_SIZE * sizeof(hiprandState));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate d_states (error code " << err << ")" << std::endl;
        return -1;
    }

    // Asynchronous memory copy to device
    err = hipMemcpyAsync(d_population, h_population, POPULATION_SIZE * GENES_PER_INDIVIDUAL * sizeof(float), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy h_population to d_population (error code " << err << ")" << std::endl;
        return -1;
    }

    // Initialize random states
    initCurandStates<<<(POPULATION_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_states, time(NULL));
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch initCurandStates kernel (error code " << err << ")" << std::endl;
        return -1;
    }

    // Synchronize before starting the genetic algorithm
    hipDeviceSynchronize();

    // Launch genetic algorithm kernel
    geneticAlgorithmKernel<<<(POPULATION_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_population, d_fitness, d_states);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch geneticAlgorithmKernel kernel (error code " << err << ")" << std::endl;
        return -1;
    }

    // Asynchronous memory copy back to host
    err = hipMemcpyAsync(h_fitness, d_fitness, POPULATION_SIZE * sizeof(float), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy d_fitness to h_fitness (error code " << err << ")" << std::endl;
        return -1;
    }

    // Synchronize to ensure all operations are complete
    hipDeviceSynchronize();

    // Free device memory
    hipFree(d_population);
    hipFree(d_fitness);
    hipFree(d_states);

    // Free host memory
    delete[] h_population;
    delete[] h_fitness;
    delete[] h_states;

    return 0;
}