#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

// Define constants
#define POPULATION_SIZE 2048
#define CHROMOSOME_LENGTH 32
#define GENERATIONS 100
#define BLOCK_SIZE 256

// Kernel to initialize the population with random values
__global__ void initializePopulation(hiprandState *states, int *population) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        // Initialize the random number generator
        hiprand_init(idx, 0, 0, &states[idx]);
        // Generate random chromosome
        for (int i = 0; i < CHROMOSOME_LENGTH; ++i) {
            population[idx * CHROMOSOME_LENGTH + i] = hiprand(&states[idx]) % 2;
        }
    }
}

// Kernel to perform genetic algorithm operations (crossover and mutation)
__global__ void geneticAlgorithmKernel(hiprandState *states, int *population, int *newPopulation) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        // Ensure uniform random selection of parents
        int parent1Idx = hiprand(&states[idx]) % POPULATION_SIZE;
        int parent2Idx = hiprand(&states[idx]) % POPULATION_SIZE;
        int crossoverPoint = hiprand(&states[idx]) % CHROMOSOME_LENGTH;

        // Perform crossover
        for (int i = 0; i < CHROMOSOME_LENGTH; ++i) {
            if (i < crossoverPoint) {
                newPopulation[idx * CHROMOSOME_LENGTH + i] = population[parent1Idx * CHROMOSOME_LENGTH + i];
            } else {
                newPopulation[idx * CHROMOSOME_LENGTH + i] = population[parent2Idx * CHROMOSOME_LENGTH + i];
            }
        }

        // Perform mutation with low probability
        for (int i = 0; i < CHROMOSOME_LENGTH; ++i) {
            if (hiprand(&states[idx]) % 100 < 1) { // 1% mutation rate
                newPopulation[idx * CHROMOSOME_LENGTH + i] = 1 - newPopulation[idx * CHROMOSOME_LENGTH + i];
            }
        }
    }
}

// Function to check for CUDA errors
void checkCudaError(hipError_t err, const char *operation) {
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << operation << " failed with error " << hipGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    int *d_population, *d_newPopulation;
    hiprandState *d_states;

    // Allocate memory on the device
    checkCudaError(hipMalloc(&d_population, POPULATION_SIZE * CHROMOSOME_LENGTH * sizeof(int)), "hipMalloc d_population");
    checkCudaError(hipMalloc(&d_newPopulation, POPULATION_SIZE * CHROMOSOME_LENGTH * sizeof(int)), "hipMalloc d_newPopulation");
    checkCudaError(hipMalloc(&d_states, POPULATION_SIZE * sizeof(hiprandState)), "hipMalloc d_states");

    // Initialize population
    int numBlocks = (POPULATION_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    initializePopulation<<<numBlocks, BLOCK_SIZE>>>(d_states, d_population);
    checkCudaError(hipGetLastError(), "initializePopulation kernel launch");
    checkCudaError(hipDeviceSynchronize(), "hipDeviceSynchronize after initializePopulation");

    // Run genetic algorithm for a number of generations
    for (int generation = 0; generation < GENERATIONS; ++generation) {
        geneticAlgorithmKernel<<<numBlocks, BLOCK_SIZE>>>(d_states, d_population, d_newPopulation);
        checkCudaError(hipGetLastError(), "geneticAlgorithmKernel kernel launch");
        checkCudaError(hipDeviceSynchronize(), "hipDeviceSynchronize after geneticAlgorithmKernel");

        // Swap populations for the next generation
        int *temp = d_population;
        d_population = d_newPopulation;
        d_newPopulation = temp;
    }

    // Copy final population back to host (for demonstration purposes)
    int *h_population = new int[POPULATION_SIZE * CHROMOSOME_LENGTH];
    checkCudaError(hipMemcpy(h_population, d_population, POPULATION_SIZE * CHROMOSOME_LENGTH * sizeof(int), hipMemcpyDeviceToHost), "hipMemcpy d_population to h_population");

    // Free device memory
    checkCudaError(hipFree(d_population), "hipFree d_population");
    checkCudaError(hipFree(d_newPopulation), "hipFree d_newPopulation");
    checkCudaError(hipFree(d_states), "hipFree d_states");

    // Clean up host memory
    delete[] h_population;

    std::cout << "Genetic algorithm completed successfully." << std::endl;
    return 0;
}