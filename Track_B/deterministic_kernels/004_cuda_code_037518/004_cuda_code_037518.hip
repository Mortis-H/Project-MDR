#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define BLOCK_SIZE 256
#define NUM_POINTS 256
#define NUM_CLUSTERS 4

__global__ void initializeRandomPoints(float* points, int numPoints, unsigned int seed) {
    hiprandState state;
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < numPoints) {
        hiprand_init(seed, id, 0, &state);
        points[id * 2] = hiprand_uniform(&state) * 100.0f;
        points[id * 2 + 1] = hiprand_uniform(&state) * 100.0f;
    }
}

__global__ void initializeRandomCentroids(float* centroids, int numClusters, unsigned int seed) {
    hiprandState state;
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < numClusters) {
        hiprand_init(seed, id, 0, &state);
        centroids[id * 2] = hiprand_uniform(&state) * 100.0f;
        centroids[id * 2 + 1] = hiprand_uniform(&state) * 100.0f;
    }
}

__global__ void assignClusters(float* points, float* centroids, int* labels, int numPoints, int numClusters) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < numPoints) {
        float minDist = 1e9f;
        int minIndex = -1;
        for (int j = 0; j < numClusters; j++) {
            float dx = points[id * 2] - centroids[j * 2];
            float dy = points[id * 2 + 1] - centroids[j * 2 + 1];
            float dist = dx * dx + dy * dy;
            if (dist < minDist) {
                minDist = dist;
                minIndex = j;
            }
        }
        labels[id] = minIndex;
    }
}

__global__ void updateCentroids(float* points, float* centroids, int* labels, int numPoints, int numClusters) {
    extern __shared__ float sharedData[];
    float* sumX = sharedData;
    float* sumY = sharedData + numClusters;
    int* count = (int*)(sharedData + 2 * numClusters);

    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int clusterId = labels[id];

    // Initialize shared memory
    if (threadIdx.x < numClusters) {
        sumX[threadIdx.x] = 0.0f;
        sumY[threadIdx.x] = 0.0f;
        count[threadIdx.x] = 0;
    }
    __syncthreads();

    // Accumulate sums and counts
    if (id < numPoints) {
        atomicAdd(&sumX[clusterId], points[id * 2]);
        atomicAdd(&sumY[clusterId], points[id * 2 + 1]);
        atomicAdd(&count[clusterId], 1);
    }
    __syncthreads();

    // Write back to global memory
    if (threadIdx.x < numClusters && count[threadIdx.x] > 0) {
        centroids[threadIdx.x * 2] = sumX[threadIdx.x] / count[threadIdx.x];
        centroids[threadIdx.x * 2 + 1] = sumY[threadIdx.x] / count[threadIdx.x];
    }
}

int main() {
    float* h_points, *h_centroids;
    int* h_labels;
    float* d_points, *d_centroids;
    int* d_labels;

    h_points = (float*)malloc(NUM_POINTS * 2 * sizeof(float));
    h_centroids = (float*)malloc(NUM_CLUSTERS * 2 * sizeof(float));
    h_labels = (int*)malloc(NUM_POINTS * sizeof(int));

    hipMalloc(&d_points, NUM_POINTS * 2 * sizeof(float));
    hipMalloc(&d_centroids, NUM_CLUSTERS * 2 * sizeof(float));
    hipMalloc(&d_labels, NUM_POINTS * sizeof(int));

    unsigned int seed = 1234;
    initializeRandomPoints<<<(NUM_POINTS + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_points, NUM_POINTS, seed);
    initializeRandomCentroids<<<(NUM_CLUSTERS + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_centroids, NUM_CLUSTERS, seed);

    hipDeviceSynchronize();

    for (int iter = 0; iter < 10; iter++) {
        assignClusters<<<(NUM_POINTS + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_points, d_centroids, d_labels, NUM_POINTS, NUM_CLUSTERS);
        updateCentroids<<<1, BLOCK_SIZE, (2 * NUM_CLUSTERS + NUM_CLUSTERS) * sizeof(float)>>>(d_points, d_centroids, d_labels, NUM_POINTS, NUM_CLUSTERS);
        hipDeviceSynchronize();
    }

    hipMemcpy(h_labels, d_labels, NUM_POINTS * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(h_centroids, d_centroids, NUM_CLUSTERS * 2 * sizeof(float), hipMemcpyDeviceToHost);

    // Output the final centroids
    std::cout << "Final centroids:" << std::endl;
    for (int i = 0; i < NUM_CLUSTERS; i++) {
        std::cout << "Centroid " << i << ": (" << h_centroids[i * 2] << ", " << h_centroids[i * 2 + 1] << ")" << std::endl;
    }

    free(h_points);
    free(h_centroids);
    free(h_labels);
    hipFree(d_points);
    hipFree(d_centroids);
    hipFree(d_labels);

    return 0;
}