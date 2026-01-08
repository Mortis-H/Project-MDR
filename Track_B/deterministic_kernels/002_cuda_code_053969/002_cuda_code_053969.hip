#include <iostream>
#include <hip/hip_runtime.h>
#include <vector>
#include <cmath>
#include <random>

#define BLOCK_SIZE 256
#define NUM_POINTS 128
#define DIMENSIONS 3
#define NUM_CLUSTERS 4

__device__ float euclidean_distance(float* point1, float* point2, int dimensions) {
    float sum = 0.0f;
    for (int i = 0; i < dimensions; ++i) {
        float diff = point1[i] - point2[i];
        sum += diff * diff;
    }
    return sqrtf(sum);
}

__global__ void assign_clusters(float* points, float* centroids, int* labels, int num_points, int num_clusters, int dimensions) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_points) {
        float min_dist = INFINITY;
        int cluster_id = -1;
        for (int j = 0; j < num_clusters; ++j) {
            float dist = euclidean_distance(&points[idx * dimensions], &centroids[j * dimensions], dimensions);
            if (dist < min_dist) {
                min_dist = dist;
                cluster_id = j;
            }
        }
        labels[idx] = cluster_id;
    }
}

__global__ void update_centroids(float* points, float* centroids, int* labels, int num_points, int num_clusters, int dimensions) {
    extern __shared__ float shared_centroids[];
    int cluster_id = threadIdx.x;
    int stride = blockDim.x;

    // Initialize shared memory
    for (int i = 0; i < dimensions; ++i) {
        shared_centroids[cluster_id * dimensions + i] = 0.0f;
    }
    __syncthreads();

    // Sum points in each cluster
    int count[NUM_CLUSTERS] = {0};
    for (int i = threadIdx.x; i < num_points; i += stride) {
        if (labels[i] == cluster_id) {
            atomicAdd(&shared_centroids[cluster_id * dimensions + 0], points[i * dimensions + 0]);
            atomicAdd(&shared_centroids[cluster_id * dimensions + 1], points[i * dimensions + 1]);
            atomicAdd(&shared_centroids[cluster_id * dimensions + 2], points[i * dimensions + 2]);
            atomicAdd(&count[cluster_id], 1);
        }
    }
    __syncthreads();

    // Write back to global memory
    if (count[cluster_id] > 0) {
        for (int i = 0; i < dimensions; ++i) {
            atomicExch(&centroids[cluster_id * dimensions + i], shared_centroids[cluster_id * dimensions + i] / count[cluster_id]);
        }
    }
}

void checkCudaError(hipError_t err, const char* msg) {
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << msg << " - " << hipGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Initialize random number generator
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(0.0f, 10.0f);

    // Allocate and initialize host memory
    std::vector<float> h_points(NUM_POINTS * DIMENSIONS);
    std::vector<float> h_centroids(NUM_CLUSTERS * DIMENSIONS);
    std::vector<int> h_labels(NUM_POINTS);

    for (auto& val : h_points) val = dis(gen);
    for (auto& val : h_centroids) val = dis(gen);

    // Allocate device memory
    float* d_points;
    float* d_centroids;
    int* d_labels;
    checkCudaError(hipMalloc(&d_points, NUM_POINTS * DIMENSIONS * sizeof(float)), "hipMalloc d_points");
    checkCudaError(hipMalloc(&d_centroids, NUM_CLUSTERS * DIMENSIONS * sizeof(float)), "hipMalloc d_centroids");
    checkCudaError(hipMalloc(&d_labels, NUM_POINTS * sizeof(int)), "hipMalloc d_labels");

    // Copy data to device
    checkCudaError(hipMemcpy(d_points, h_points.data(), NUM_POINTS * DIMENSIONS * sizeof(float), hipMemcpyHostToDevice), "hipMemcpy d_points");
    checkCudaError(hipMemcpy(d_centroids, h_centroids.data(), NUM_CLUSTERS * DIMENSIONS * sizeof(float), hipMemcpyHostToDevice), "hipMemcpy d_centroids");

    // K-means iterations
    int max_iterations = 100;
    for (int iter = 0; iter < max_iterations; ++iter) {
        assign_clusters<<<(NUM_POINTS + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_points, d_centroids, d_labels, NUM_POINTS, NUM_CLUSTERS, DIMENSIONS);
        checkCudaError(hipGetLastError(), "assign_clusters kernel launch");

        update_centroids<<<NUM_CLUSTERS, BLOCK_SIZE, NUM_CLUSTERS * DIMENSIONS * sizeof(float)>>>(d_points, d_centroids, d_labels, NUM_POINTS, NUM_CLUSTERS, DIMENSIONS);
        checkCudaError(hipGetLastError(), "update_centroids kernel launch");

        hipDeviceSynchronize();
    }

    // Copy results back to host
    checkCudaError(hipMemcpy(h_labels.data(), d_labels, NUM_POINTS * sizeof(int), hipMemcpyDeviceToHost), "hipMemcpy h_labels");

    // Free device memory
    hipFree(d_points);
    hipFree(d_centroids);
    hipFree(d_labels);

    return 0;
}