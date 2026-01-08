// genetic_algorithm.cu
#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define POPULATION_SIZE 32
#define GENOME_LENGTH 128
#define NUM_GENERATIONS 100

// Kernel to initialize the population with random values
__global__ void initializePopulation(hiprandState *state, unsigned int *population) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        hiprand_init(idx, 0, 0, &state[idx]);
        for (int i = 0; i < GENOME_LENGTH; ++i) {
            population[idx * GENOME_LENGTH + i] = hiprand(&state[idx]) % 2;
        }
    }
}

// Kernel to evaluate fitness of each genome
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

// Kernel to perform selection, crossover, and mutation
__global__ void evolvePopulation(hiprandState *state, unsigned int *population, float *fitness) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        hiprandState localState = state[idx];
        // Simple roulette wheel selection
        float sumFitness = 0.0f;
        for (int i = 0; i < POPULATION_SIZE; ++i) {
            sumFitness += fitness[i];
        }
        float randSelect = hiprand_uniform(&localState) * sumFitness;
        int parent1 = 0;
        float cumulativeFitness = 0.0f;
        while (cumulativeFitness < randSelect && parent1 < POPULATION_SIZE) {
            cumulativeFitness += fitness[parent1];
            parent1++;
        }
        parent1 = (parent1 > 0) ? parent1 - 1 : 0;

        randSelect = hiprand_uniform(&localState) * sumFitness;
        int parent2 = 0;
        cumulativeFitness = 0.0f;
        while (cumulativeFitness < randSelect && parent2 < POPULATION_SIZE) {
            cumulativeFitness += fitness[parent2];
            parent2++;
        }
        parent2 = (parent2 > 0) ? parent2 - 1 : 0;

        // Single-point crossover
        int crossoverPoint = hiprand(&localState) % GENOME_LENGTH;
        for (int i = 0; i < GENOME_LENGTH; ++i) {
            if (i < crossoverPoint) {
                population[idx * GENOME_LENGTH + i] = population[parent1 * GENOME_LENGTH + i];
            } else {
                population[idx * GENOME_LENGTH + i] = population[parent2 * GENOME_LENGTH + i];
            }
        }

        // Mutation
        for (int i = 0; i < GENOME_LENGTH; ++i) {
            if (hiprand_uniform(&localState) < 0.01f) {
                population[idx * GENOME_LENGTH + i] = 1 - population[idx * GENOME_LENGTH + i];
            }
        }
        state[idx] = localState;
    }
}

int main() {
    unsigned int *d_population;
    float *d_fitness;
    hiprandState *d_state;

    // Allocate memory on the device
    hipError_t err = hipMalloc((void**)&d_population, POPULATION_SIZE * GENOME_LENGTH * sizeof(unsigned int));
    if (err != hipSuccess) {
        std::cerr << "Error allocating population: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc((void**)&d_fitness, POPULATION_SIZE * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Error allocating fitness: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc((void**)&d_state, POPULATION_SIZE * sizeof(hiprandState));
    if (err != hipSuccess) {
        std::cerr << "Error allocating state: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Initialize population
    initializePopulation<<<1, POPULATION_SIZE>>>(d_state, d_population);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Error launching initializePopulation: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    hipDeviceSynchronize();

    // Genetic algorithm loop
    for (int generation = 0; generation < NUM_GENERATIONS; ++generation) {
        // Evaluate fitness
        evaluateFitness<<<1, POPULATION_SIZE>>>(d_population, d_fitness);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Error launching evaluateFitness: " << hipGetErrorString(err) << std::endl;
            return -1;
        }
        hipDeviceSynchronize();

        // Evolve population
        evolvePopulation<<<1, POPULATION_SIZE>>>(d_state, d_population, d_fitness);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Error launching evolvePopulation: " << hipGetErrorString(err) << std::endl;
            return -1;
        }
        hipDeviceSynchronize();
    }

    // Free device memory
    hipFree(d_population);
    hipFree(d_fitness);
    hipFree(d_state);

    return 0;
}