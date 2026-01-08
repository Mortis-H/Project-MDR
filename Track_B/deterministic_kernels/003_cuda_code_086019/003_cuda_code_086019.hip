#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define POPULATION_SIZE 32
#define GENOME_LENGTH 128
#define NUM_GENERATIONS 100

__global__ void initializePopulation(hiprandState *states, float *population) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < POPULATION_SIZE) {
        hiprand_init(idx, 0, 0, &states[idx]);
        for (int i = 0; i < GENOME_LENGTH; ++i) {
            population[idx * GENOME_LENGTH + i] = hiprand_uniform(&states[idx]);
        }
    }
}

__global__ void evaluateFitness(float *population, float *fitness) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < POPULATION_SIZE) {
        float sum = 0.0f;
        for (int i = 0; i < GENOME_LENGTH; ++i) {
            sum += population[idx * GENOME_LENGTH + i];
        }
        fitness[idx] = sum / GENOME_LENGTH; // Simple fitness function
    }
}

__global__ void crossover(float *population, float *new_population, float *fitness) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < POPULATION_SIZE) {
        int parent1 = idx;
        int parent2 = (idx + 1) % POPULATION_SIZE;
        if (fitness[parent1] < fitness[parent2]) {
            int temp = parent1;
            parent1 = parent2;
            parent2 = temp;
        }
        for (int i = 0; i < GENOME_LENGTH; ++i) {
            new_population[idx * GENOME_LENGTH + i] = population[parent1 * GENOME_LENGTH + i];
        }
    }
}

int main() {
    float *h_population, *h_new_population, *h_fitness;
    float *d_population, *d_new_population, *d_fitness;
    hiprandState *d_states;

    // Allocate host memory
    h_population = (float *)malloc(POPULATION_SIZE * GENOME_LENGTH * sizeof(float));
    h_new_population = (float *)malloc(POPULATION_SIZE * GENOME_LENGTH * sizeof(float));
    h_fitness = (float *)malloc(POPULATION_SIZE * sizeof(float));

    // Allocate device memory
    hipError_t err = hipMalloc((void **)&d_population, POPULATION_SIZE * GENOME_LENGTH * sizeof(float));
    if (err != hipSuccess) { std::cerr << "Error allocating memory for d_population: " << hipGetErrorString(err) << std::endl; return -1; }
    err = hipMalloc((void **)&d_new_population, POPULATION_SIZE * GENOME_LENGTH * sizeof(float));
    if (err != hipSuccess) { std::cerr << "Error allocating memory for d_new_population: " << hipGetErrorString(err) << std::endl; return -1; }
    err = hipMalloc((void **)&d_fitness, POPULATION_SIZE * sizeof(float));
    if (err != hipSuccess) { std::cerr << "Error allocating memory for d_fitness: " << hipGetErrorString(err) << std::endl; return -1; }
    err = hipMalloc((void **)&d_states, POPULATION_SIZE * sizeof(hiprandState));
    if (err != hipSuccess) { std::cerr << "Error allocating memory for d_states: " << hipGetErrorString(err) << std::endl; return -1; }

    // Initialize population on GPU
    initializePopulation<<<(POPULATION_SIZE + 255) / 256, 256>>>(d_states, d_population);
    err = hipGetLastError();
    if (err != hipSuccess) { std::cerr << "Error launching initializePopulation kernel: " << hipGetErrorString(err) << std::endl; return -1; }

    // Main genetic algorithm loop
    for (int generation = 0; generation < NUM_GENERATIONS; ++generation) {
        // Asynchronously copy population to device
        err = hipMemcpyAsync(d_population, h_population, POPULATION_SIZE * GENOME_LENGTH * sizeof(float), hipMemcpyHostToDevice, 0);
        if (err != hipSuccess) { std::cerr << "Error copying population to device: " << hipGetErrorString(err) << std::endl; return -1; }

        // Evaluate fitness
        evaluateFitness<<<(POPULATION_SIZE + 255) / 256, 256>>>(d_population, d_fitness);
        err = hipGetLastError();
        if (err != hipSuccess) { std::cerr << "Error launching evaluateFitness kernel: " << hipGetErrorString(err) << std::endl; return -1; }

        // Crossover
        crossover<<<(POPULATION_SIZE + 255) / 256, 256>>>(d_population, d_new_population, d_fitness);
        err = hipGetLastError();
        if (err != hipSuccess) { std::cerr << "Error launching crossover kernel: " << hipGetErrorString(err) << std::endl; return -1; }

        // Asynchronously copy new population to host
        err = hipMemcpyAsync(h_new_population, d_new_population, POPULATION_SIZE * GENOME_LENGTH * sizeof(float), hipMemcpyDeviceToHost, 0);
        if (err != hipSuccess) { std::cerr << "Error copying new population to host: " << hipGetErrorString(err) << std::endl; return -1; }

        // Synchronize to ensure all operations are complete
        hipDeviceSynchronize();

        // Swap population pointers for the next generation
        float *temp = h_population;
        h_population = h_new_population;
        h_new_population = temp;
    }

    // Copy final population back to host
    err = hipMemcpy(h_population, d_population, POPULATION_SIZE * GENOME_LENGTH * sizeof(float), hipMemcpyDeviceToHost);
    if (err != hipSuccess) { std::cerr << "Error copying final population to host: " << hipGetErrorString(err) << std::endl; return -1; }

    // Free device memory
    hipFree(d_population);
    hipFree(d_new_population);
    hipFree(d_fitness);
    hipFree(d_states);

    // Free host memory
    free(h_population);
    free(h_new_population);
    free(h_fitness);

    return 0;
}