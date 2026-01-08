#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define POPULATION_SIZE 4096
#define GENOME_LENGTH 128
#define BLOCK_SIZE 256

// Kernel to initialize random genomes
__global__ void initializePopulation(hiprandState *states, unsigned int *population) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        hiprand_init(idx, 0, 0, &states[idx]);
        for (int i = 0; i < GENOME_LENGTH; ++i) {
            population[idx * GENOME_LENGTH + i] = hiprand(&states[idx]) % 2;
        }
    }
}

// Kernel to evaluate fitness (simple example: count number of 1s)
__global__ void evaluateFitness(unsigned int *population, float *fitness) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        float fit = 0.0f;
        for (int i = 0; i < GENOME_LENGTH; ++i) {
            fit += population[idx * GENOME_LENGTH + i];
        }
        fitness[idx] = fit / GENOME_LENGTH;
    }
}

// Kernel to perform crossover and mutation
__global__ void crossoverAndMutate(hiprandState *states, unsigned int *population, float *fitness) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        hiprandState localState = states[idx];
        int parent1 = idx;
        int parent2 = hiprand(&localState) % POPULATION_SIZE;
        float fit1 = fitness[parent1];
        float fit2 = fitness[parent2];
        int betterParent = (fit1 > fit2) ? parent1 : parent2;

        for (int i = 0; i < GENOME_LENGTH; ++i) {
            if (hiprand(&localState) < 0.5f) {
                population[idx * GENOME_LENGTH + i] = population[betterParent * GENOME_LENGTH + i];
            }
            if (hiprand(&localState) < 0.01f) { // Mutation rate
                population[idx * GENOME_LENGTH + i] = 1 - population[idx * GENOME_LENGTH + i];
            }
        }
        states[idx] = localState;
    }
}

int main() {
    unsigned int *d_population;
    float *d_fitness;
    hiprandState *d_states;

    // Allocate memory on device
    hipError_t err = hipMalloc((void**)&d_population, POPULATION_SIZE * GENOME_LENGTH * sizeof(unsigned int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate population: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc((void**)&d_fitness, POPULATION_SIZE * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate fitness: " << hipGetErrorString(err) << std::endl;
        hipFree(d_population);
        return -1;
    }
    err = hipMalloc((void**)&d_states, POPULATION_SIZE * sizeof(hiprandState));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate states: " << hipGetErrorString(err) << std::endl;
        hipFree(d_population);
        hipFree(d_fitness);
        return -1;
    }

    // Initialize population
    initializePopulation<<<(POPULATION_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_states, d_population);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch initializePopulation: " << hipGetErrorString(err) << std::endl;
        hipFree(d_population);
        hipFree(d_fitness);
        hipFree(d_states);
        return -1;
    }
    hipDeviceSynchronize();

    // Evaluate fitness
    evaluateFitness<<<(POPULATION_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_population, d_fitness);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch evaluateFitness: " << hipGetErrorString(err) << std::endl;
        hipFree(d_population);
        hipFree(d_fitness);
        hipFree(d_states);
        return -1;
    }
    hipDeviceSynchronize();

    // Perform crossover and mutation
    crossoverAndMutate<<<(POPULATION_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_states, d_population, d_fitness);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch crossoverAndMutate: " << hipGetErrorString(err) << std::endl;
        hipFree(d_population);
        hipFree(d_fitness);
        hipFree(d_states);
        return -1;
    }
    hipDeviceSynchronize();

    // Free device memory
    hipFree(d_population);
    hipFree(d_fitness);
    hipFree(d_states);

    return 0;
}