#include <iostream>
#include <hip/hip_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/scan.h>
#include <thrust/unique.h>
#include <thrust/sequence.h>
#include <thrust/fill.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>

#define BLOCK_SIZE 256

__device__ int find(int *parent, int x) {
    while (x != parent[x]) {
        x = parent[x];
    }
    return x;
}

__device__ void unite(int *parent, int *rank, int x, int y) {
    int rootX = find(parent, x);
    int rootY = find(parent, y);
    if (rootX != rootY) {
        if (rank[rootX] > rank[rootY]) {
            parent[rootY] = rootX;
        } else if (rank[rootX] < rank[rootY]) {
            parent[rootX] = rootY;
        } else {
            parent[rootY] = rootX;
            rank[rootX]++;
        }
    }
}

__global__ void initializeParents(int *parent, int *rank, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        parent[idx] = idx;
        rank[idx] = 0;
    }
}

__global__ void findWatershedLabels(int *image, int *parent, int *rank, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = y * width + x;
        int minVal = image[idx];
        int minIdx = idx;

        // Check neighbors for minimum value
        if (x > 0 && image[idx - 1] < minVal) {
            minVal = image[idx - 1];
            minIdx = idx - 1;
        }
        if (x < width - 1 && image[idx + 1] < minVal) {
            minVal = image[idx + 1];
            minIdx = idx + 1;
        }
        if (y > 0 && image[idx - width] < minVal) {
            minVal = image[idx - width];
            minIdx = idx - width;
        }
        if (y < height - 1 && image[idx + width] < minVal) {
            minVal = image[idx + width];
            minIdx = idx + width;
        }

        // Unite with the neighbor having the minimum value
        unite(parent, rank, idx, minIdx);
    }
}

__global__ void assignFinalLabels(int *image, int *parent, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = y * width + x;
        image[idx] = find(parent, idx);
    }
}

int main() {
    int width = 1024;
    int height = 1024;
    int size = width * height;

    // Allocate host memory
    thrust::host_vector<int> h_image(size);
    thrust::host_vector<int> h_parent(size);
    thrust::host_vector<int> h_rank(size);

    // Initialize image with some values (example: gradient)
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            h_image[i * width + j] = abs(i - j); // Simple gradient
        }
    }

    // Allocate device memory
    int *d_image, *d_parent, *d_rank;
    hipMalloc(&d_image, size * sizeof(int));
    hipMalloc(&d_parent, size * sizeof(int));
    hipMalloc(&d_rank, size * sizeof(int));

    // Copy data from host to device
    hipMemcpy(d_image, h_image.data(), size * sizeof(int), hipMemcpyHostToDevice);

    // Initialize parent and rank arrays
    dim3 threadsPerBlock(BLOCK_SIZE);
    dim3 numBlocks((size + threadsPerBlock.x - 1) / threadsPerBlock.x);
    initializeParents<<<numBlocks, threadsPerBlock>>>(d_parent, d_rank, size);
    hipDeviceSynchronize();

    // Find watershed labels
    numBlocks = dim3((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    findWatershedLabels<<<numBlocks, threadsPerBlock>>>(d_image, d_parent, d_rank, width, height);
    hipDeviceSynchronize();

    // Assign final labels
    assignFinalLabels<<<numBlocks, threadsPerBlock>>>(d_image, d_parent, width, height);
    hipDeviceSynchronize();

    // Copy result back to host
    hipMemcpy(h_image.data(), d_image, size * sizeof(int), hipMemcpyDeviceToHost);

    // Free device memory
    hipFree(d_image);
    hipFree(d_parent);
    hipFree(d_rank);

    // Check for errors
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    std::cout << "Watershed segmentation completed successfully." << std::endl;
    return 0;
}