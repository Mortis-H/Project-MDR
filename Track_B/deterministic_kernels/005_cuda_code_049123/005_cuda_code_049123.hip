// genetic_algorithm.cu
#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define POPULATION_SIZE 512
#define GENES 16
#define NUM_THREADS 256

// Kernel to initialize the population with random values
__global__ void initializePopulation(hiprandState *state, int *population) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        hiprand_init((unsigned long long)clock() + idx, 0, 0, &state[idx]);
        for (int i = 0; i < GENES; ++i) {
            population[idx * GENES + i] = hiprand(&state[idx]) % 2;
        }
    }
}

// Kernel to evaluate fitness (simple example: count number of 1s)
__global__ void evaluateFitness(int *population, int *fitness) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        int fit = 0;
        for (int i = 0; i < GENES; ++i) {
            fit += population[idx * GENES + i];
        }
        fitness[idx] = fit;
    }
}

// Kernel to perform selection (simple example: copy fittest individual)
__global__ void selection(int *population, int *fitness, int *newPopulation) {
    extern __shared__ int s_fitness[];
    extern __shared__ int s_population[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx < POPULATION_SIZE) {
        s_fitness[tid] = fitness[idx];
        for (int i = 0; i < GENES; ++i) {
            s_population[tid * GENES + i] = population[idx * GENES + i];
        }
        __syncthreads();

        int maxFit = s_fitness[0];
        int maxIdx = 0;
        for (int i = 1; i < blockDim.x; ++i) {
            if (s_fitness[i] > maxFit) {
                maxFit = s_fitness[i];
                maxIdx = i;
            }
        }
        __syncthreads();

        if (tid == 0) {
            for (int i = 0; i < GENES; ++i) {
                newPopulation[blockIdx.x * blockDim.x * GENES + tid * GENES + i] = s_population[maxIdx * GENES + i];
            }
        }
    }
}

// Kernel to perform crossover (simple example: one-point crossover)
__global__ void crossover(int *population, int *newPopulation) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        int partnerIdx = (idx + 1) % POPULATION_SIZE;
        int crossoverPoint = GENES / 2;
        for (int i = 0; i < crossoverPoint; ++i) {
            newPopulation[idx * GENES + i] = population[idx * GENES + i];
        }
        for (int i = crossoverPoint; i < GENES; ++i) {
            newPopulation[idx * GENES + i] = population[partnerIdx * GENES + i];
        }
    }
}

// Kernel to perform mutation (simple example: flip a bit with a small probability)
__global__ void mutation(hiprandState *state, int *population) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POPULATION_SIZE) {
        for (int i = 0; i < GENES; ++i) {
            if (hiprand(&state[idx]) % 100 < 5) { // 5% mutation rate
                population[idx * GENES + i] = 1 - population[idx * GENES + i];
            }
        }
    }
}

int main() {
    int *d_population, *d_newPopulation, *d_fitness;
    hiprandState *d_state;

    // Allocate memory
    hipError_t err = hipMalloc((void**)&d_population, POPULATION_SIZE * GENES * sizeof(int));
    if (err != hipSuccess) { std::cerr << "Error allocating memory for population\n"; return 1; }
    err = hipMalloc((void**)&d_newPopulation, POPULATION_SIZE * GENES * sizeof(int));
    if (err != hipSuccess) { std::cerr << "Error allocating memory for new population\n"; return 1; }
    err = hipMalloc((void**)&d_fitness, POPULATION_SIZE * sizeof(int));
    if (err != hipSuccess) { std::cerr << "Error allocating memory for fitness\n"; return 1; }
    err = hipMalloc((void**)&d_state, POPULATION_SIZE * sizeof(hiprandState));
    if (err != hipSuccess) { std::cerr << "Error allocating memory for random states\n"; return 1; }

    // Initialize population
    initializePopulation<<<(POPULATION_SIZE + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(d_state, d_population);
    hipDeviceSynchronize();
    if (err != hipSuccess) { std::cerr << "Error initializing population\n"; return 1; }

    // Main genetic algorithm loop
    for (int generation = 0; generation < 10; ++generation) {
        // Evaluate fitness
        evaluateFitness<<<(POPULATION_SIZE + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(d_population, d_fitness);
        hipDeviceSynchronize();
        if (err != hipSuccess) { std::cerr << "Error evaluating fitness\n"; return 1; }

        // Selection
        selection<<<(POPULATION_SIZE + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS, (GENES + 1) * sizeof(int)>>>(d_population, d_fitness, d_newPopulation);
        hipDeviceSynchronize();
        if (err != hipSuccess) { std::cerr << "Error performing selection\n"; return 1; }

        // Crossover
        crossover<<<(POPULATION_SIZE + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(d_newPopulation, d_population);
        hipDeviceSynchronize();
        if (err != hipSuccess) { std::cerr << "Error performing crossover\n"; return 1; }

        // Mutation
        mutation<<<(POPULATION_SIZE + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(d_state, d_population);
        hipDeviceSynchronize();
        if (err != hipSuccess) { std::cerr << "Error performing mutation\n"; return 1; }
    }

    // Free memory
    hipFree(d_population);
    hipFree(d_newPopulation);
    hipFree(d_fitness);
    hipFree(d_state);

    return 0;
}