#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define POPULATION_SIZE 1024
#define VARIABLES 8192
#define BLOCK_SIZE 256

__global__ void initializePopulation(hiprandState *states, float *population) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        for (int i = 0; i < VARIABLES; ++i) {
            population[idx * VARIABLES + i] = hiprand_uniform(&states[idx]);
        }
    }
}

__device__ float fitnessFunction(const float *individual) {
    float fitness = 0.0f;
    for (int i = 0; i < VARIABLES; ++i) {
        fitness += individual[i] * individual[i]; // Example: Sum of squares
    }
    return fitness;
}

__global__ void evaluateFitness(const float *population, float *fitness) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        fitness[idx] = fitnessFunction(&population[idx * VARIABLES]);
    }
}

__global__ void crossover(const float *population, float *newPopulation, float *fitness, hiprandState *states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        int parent1 = idx;
        int parent2 = (idx + 1) % POPULATION_SIZE;
        int crossoverPoint = hiprand(&states[idx]) % VARIABLES;

        for (int i = 0; i < crossoverPoint; ++i) {
            newPopulation[idx * VARIABLES + i] = population[parent1 * VARIABLES + i];
        }
        for (int i = crossoverPoint; i < VARIABLES; ++i) {
            newPopulation[idx * VARIABLES + i] = population[parent2 * VARIABLES + i];
        }
    }
}

__global__ void mutate(float *population, hiprandState *states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        for (int i = 0; i < VARIABLES; ++i) {
            if (hiprand_uniform(&states[idx]) < 0.01f) { // Mutation rate of 1%
                population[idx * VARIABLES + i] += (hiprand_uniform(&states[idx]) - 0.5f) * 0.1f; // Small random change
            }
        }
    }
}

__global__ void initializeCurandStates(hiprandState *states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        hiprand_init(seed, idx, 0, &states[idx]);
    }
}

int main() {
    float *d_population, *d_newPopulation, *d_fitness;
    hiprandState *d_states;
    float *h_population, *h_fitness;

    h_population = (float *)malloc(POPULATION_SIZE * VARIABLES * sizeof(float));
    h_fitness = (float *)malloc(POPULATION_SIZE * sizeof(float));

    hipMalloc(&d_population, POPULATION_SIZE * VARIABLES * sizeof(float));
    hipMalloc(&d_newPopulation, POPULATION_SIZE * VARIABLES * sizeof(float));
    hipMalloc(&d_fitness, POPULATION_SIZE * sizeof(float));
    hipMalloc(&d_states, POPULATION_SIZE * sizeof(hiprandState));

    int numBlocks = (POPULATION_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

    initializeCurandStates<<<numBlocks, BLOCK_SIZE>>>(d_states, 1234);
    hipDeviceSynchronize();

    initializePopulation<<<numBlocks, BLOCK_SIZE>>>(d_states, d_population);
    hipDeviceSynchronize();

    for (int generation = 0; generation < 100; ++generation) {
        evaluateFitness<<<numBlocks, BLOCK_SIZE>>>(d_population, d_fitness);
        hipDeviceSynchronize();

        crossover<<<numBlocks, BLOCK_SIZE>>>(d_population, d_newPopulation, d_fitness, d_states);
        hipDeviceSynchronize();

        mutate<<<numBlocks, BLOCK_SIZE>>>(d_newPopulation, d_states);
        hipDeviceSynchronize();

        hipMemcpy(d_population, d_newPopulation, POPULATION_SIZE * VARIABLES * sizeof(float), hipMemcpyDeviceToDevice);
    }

    hipMemcpy(h_fitness, d_fitness, POPULATION_SIZE * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(h_population, d_population, POPULATION_SIZE * VARIABLES * sizeof(float), hipMemcpyDeviceToHost);

    float minFitness = h_fitness[0];
    int bestIndex = 0;
    for (int i = 1; i < POPULATION_SIZE; ++i) {
        if (h_fitness[i] < minFitness) {
            minFitness = h_fitness[i];
            bestIndex = i;
        }
    }

    std::cout << "Best fitness: " << minFitness << std::endl;
    std::cout << "Best individual: ";
    for (int i = 0; i < VARIABLES; ++i) {
        std::cout << h_population[bestIndex * VARIABLES + i] << " ";
    }
    std::cout << std::endl;

    free(h_population);
    free(h_fitness);
    hipFree(d_population);
    hipFree(d_newPopulation);
    hipFree(d_fitness);
    hipFree(d_states);

    return 0;
}