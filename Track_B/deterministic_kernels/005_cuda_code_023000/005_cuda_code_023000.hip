#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define POPULATION_SIZE 1024
#define NUM_VARIABLES 64
#define NUM_GENERATIONS 1000
#define BLOCK_SIZE 256

// Kernel to initialize the population with random values
__global__ void initializePopulation(hiprandState *states, float *population) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        hiprand_init((unsigned long long)clock() + idx, 0, 0, &states[idx]);
        for (int i = 0; i < NUM_VARIABLES; i++) {
            population[idx * NUM_VARIABLES + i] = hiprand_uniform(&states[idx]) * 100.0f;
        }
    }
}

// Kernel to evaluate fitness (simple example: sum of squares)
__global__ void evaluateFitness(float *population, float *fitness) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        float sum = 0.0f;
        for (int i = 0; i < NUM_VARIABLES; i++) {
            sum += population[idx * NUM_VARIABLES + i] * population[idx * NUM_VARIABLES + i];
        }
        fitness[idx] = sum;
    }
}

// Kernel to perform selection (simple tournament selection)
__global__ void selectParents(float *fitness, int *parents) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        int parent1 = idx;
        int parent2 = (idx + 1) % POPULATION_SIZE;
        parents[idx * 2] = (fitness[parent1] < fitness[parent2]) ? parent1 : parent2;
        parents[idx * 2 + 1] = (fitness[parent1] >= fitness[parent2]) ? parent1 : parent2;
    }
}

// Kernel to perform crossover (simple one-point crossover)
__global__ void crossover(float *population, int *parents, float *newPopulation) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        int parent1 = parents[idx * 2];
        int parent2 = parents[idx * 2 + 1];
        int crossoverPoint = idx % NUM_VARIABLES;
        for (int i = 0; i < NUM_VARIABLES; i++) {
            if (i < crossoverPoint) {
                newPopulation[idx * NUM_VARIABLES + i] = population[parent1 * NUM_VARIABLES + i];
            } else {
                newPopulation[idx * NUM_VARIABLES + i] = population[parent2 * NUM_VARIABLES + i];
            }
        }
    }
}

// Kernel to perform mutation (simple random mutation)
__global__ void mutate(hiprandState *states, float *population) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        hiprandState state = states[idx];
        for (int i = 0; i < NUM_VARIABLES; i++) {
            if (hiprand_uniform(&state) < 0.01f) { // Mutation rate
                population[idx * NUM_VARIABLES + i] += hiprand_uniform(&state) * 10.0f - 5.0f;
            }
        }
        states[idx] = state;
    }
}

int main() {
    float *h_population, *h_newPopulation, *h_fitness;
    float *d_population, *d_newPopulation, *d_fitness;
    hiprandState *d_states;
    int *d_parents;

    // Allocate host memory
    h_population = (float *)malloc(POPULATION_SIZE * NUM_VARIABLES * sizeof(float));
    h_newPopulation = (float *)malloc(POPULATION_SIZE * NUM_VARIABLES * sizeof(float));
    h_fitness = (float *)malloc(POPULATION_SIZE * sizeof(float));

    // Allocate device memory
    hipMalloc((void **)&d_population, POPULATION_SIZE * NUM_VARIABLES * sizeof(float));
    hipMalloc((void **)&d_newPopulation, POPULATION_SIZE * NUM_VARIABLES * sizeof(float));
    hipMalloc((void **)&d_fitness, POPULATION_SIZE * sizeof(float));
    hipMalloc((void **)&d_states, POPULATION_SIZE * sizeof(hiprandState));
    hipMalloc((void **)&d_parents, POPULATION_SIZE * 2 * sizeof(int));

    // Initialize population on GPU
    initializePopulation<<<(POPULATION_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_states, d_population);
    hipDeviceSynchronize();

    // Main genetic algorithm loop
    for (int generation = 0; generation < NUM_GENERATIONS; generation++) {
        // Evaluate fitness
        evaluateFitness<<<(POPULATION_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_population, d_fitness);
        hipDeviceSynchronize();

        // Select parents
        selectParents<<<(POPULATION_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_fitness, d_parents);
        hipDeviceSynchronize();

        // Crossover
        crossover<<<(POPULATION_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_population, d_parents, d_newPopulation);
        hipDeviceSynchronize();

        // Mutate
        mutate<<<(POPULATION_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_states, d_newPopulation);
        hipDeviceSynchronize();

        // Swap populations
        float *temp = d_population;
        d_population = d_newPopulation;
        d_newPopulation = temp;
    }

    // Copy final population back to host
    hipMemcpy(h_population, d_population, POPULATION_SIZE * NUM_VARIABLES * sizeof(float), hipMemcpyDeviceToHost);

    // Free device memory
    hipFree(d_population);
    hipFree(d_newPopulation);
    hipFree(d_fitness);
    hipFree(d_states);
    hipFree(d_parents);

    // Free host memory
    free(h_population);
    free(h_newPopulation);
    free(h_fitness);

    std::cout << "Genetic algorithm completed successfully." << std::endl;

    return 0;
}