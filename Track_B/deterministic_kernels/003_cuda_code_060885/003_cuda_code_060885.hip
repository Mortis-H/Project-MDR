#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

// Define constants
#define POPULATION_SIZE 4096
#define GENES_PER_INDIVIDUAL 16
#define BLOCK_SIZE 256

// Kernel to initialize the population with random genes
__global__ void initializePopulation(hiprandState *states, int *population) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        // Initialize curand state
        hiprand_init(idx, 0, 0, &states[idx]);
        // Generate random genes for each individual
        for (int i = 0; i < GENES_PER_INDIVIDUAL; ++i) {
            population[idx * GENES_PER_INDIVIDUAL + i] = hiprand(&states[idx]) % 2;
        }
    }
}

// Kernel to perform crossover operation between pairs of individuals
__global__ void crossover(int *population, int *newPopulation, int crossoverPoint) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE / 2) {
        int parent1Idx = idx * 2;
        int parent2Idx = idx * 2 + 1;
        int child1Idx = idx * 2;
        int child2Idx = idx * 2 + 1;

        // Perform crossover
        for (int i = 0; i < GENES_PER_INDIVIDUAL; ++i) {
            if (i < crossoverPoint) {
                newPopulation[child1Idx * GENES_PER_INDIVIDUAL + i] = population[parent1Idx * GENES_PER_INDIVIDUAL + i];
                newPopulation[child2Idx * GENES_PER_INDIVIDUAL + i] = population[parent2Idx * GENES_PER_INDIVIDUAL + i];
            } else {
                newPopulation[child1Idx * GENES_PER_INDIVIDUAL + i] = population[parent2Idx * GENES_PER_INDIVIDUAL + i];
                newPopulation[child2Idx * GENES_PER_INDIVIDUAL + i] = population[parent1Idx * GENES_PER_INDIVIDUAL + i];
            }
        }
    }
}

// Kernel to perform mutation operation on the population
__global__ void mutate(int *population, float mutationRate, hiprandState *states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        for (int i = 0; i < GENES_PER_INDIVIDUAL; ++i) {
            float r = hiprand_uniform(&states[idx]);
            if (r < mutationRate) {
                population[idx * GENES_PER_INDIVIDUAL + i] = 1 - population[idx * GENES_PER_INDIVIDUAL + i];
            }
        }
    }
}

int main() {
    int *d_population, *d_newPopulation;
    hiprandState *d_states;
    size_t populationSizeBytes = POPULATION_SIZE * GENES_PER_INDIVIDUAL * sizeof(int);
    size_t statesSizeBytes = POPULATION_SIZE * sizeof(hiprandState);

    // Allocate memory on the device
    hipError_t err = hipMalloc(&d_population, populationSizeBytes);
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate memory for population (error code " << err << ")" << std::endl;
        return -1;
    }
    err = hipMalloc(&d_newPopulation, populationSizeBytes);
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate memory for new population (error code " << err << ")" << std::endl;
        hipFree(d_population);
        return -1;
    }
    err = hipMalloc(&d_states, statesSizeBytes);
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate memory for states (error code " << err << ")" << std::endl;
        hipFree(d_population);
        hipFree(d_newPopulation);
        return -1;
    }

    // Initialize population on the device
    int blocksPerGrid = (POPULATION_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    initializePopulation<<<blocksPerGrid, BLOCK_SIZE>>>(d_states, d_population);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch initializePopulation kernel (error code " << err << ")" << std::endl;
        hipFree(d_population);
        hipFree(d_newPopulation);
        hipFree(d_states);
        return -1;
    }
    hipDeviceSynchronize();

    // Perform crossover
    crossover<<<blocksPerGrid / 2, BLOCK_SIZE>>>(d_population, d_newPopulation, GENES_PER_INDIVIDUAL / 2);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch crossover kernel (error code " << err << ")" << std::endl;
        hipFree(d_population);
        hipFree(d_newPopulation);
        hipFree(d_states);
        return -1;
    }
    hipDeviceSynchronize();

    // Perform mutation
    float mutationRate = 0.01f;
    mutate<<<blocksPerGrid, BLOCK_SIZE>>>(d_newPopulation, mutationRate, d_states);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch mutate kernel (error code " << err << ")" << std::endl;
        hipFree(d_population);
        hipFree(d_newPopulation);
        hipFree(d_states);
        return -1;
    }
    hipDeviceSynchronize();

    // Copy result back to host for verification (optional)
    int *h_population = new int[POPULATION_SIZE * GENES_PER_INDIVIDUAL];
    err = hipMemcpy(h_population, d_newPopulation, populationSizeBytes, hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy population from device to host (error code " << err << ")" << std::endl;
        hipFree(d_population);
        hipFree(d_newPopulation);
        hipFree(d_states);
        delete[] h_population;
        return -1;
    }

    // Free device memory
    hipFree(d_population);
    hipFree(d_newPopulation);
    hipFree(d_states);

    // Clean up host memory
    delete[] h_population;

    std::cout << "Genetic algorithm operations completed successfully." << std::endl;
    return 0;
}