#include <iostream>
#include <hip/hip_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#define BLOCK_SIZE 256

__device__ int find(int *parent, int x) {
    while (parent[x] != x) {
        parent[x] = parent[parent[x]]; // Path compression
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

__global__ void initializeLabels(int *labels, int *parent, int *rank, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width * height) {
        labels[idx] = idx;
        parent[idx] = idx;
        rank[idx] = 0;
    }
}

__global__ void findMinima(int *labels, int *parent, int *rank, int *minima, int *image, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int x = idx % width;
    int y = idx / width;
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        int minVal = image[idx];
        int minIdx = idx;
        if (image[idx - 1] < minVal) {
            minVal = image[idx - 1];
            minIdx = idx - 1;
        }
        if (image[idx + 1] < minVal) {
            minVal = image[idx + 1];
            minIdx = idx + 1;
        }
        if (image[idx - width] < minVal) {
            minVal = image[idx - width];
            minIdx = idx - width;
        }
        if (image[idx + width] < minVal) {
            minVal = image[idx + width];
            minIdx = idx + width;
        }
        minima[idx] = minIdx;
        if (minIdx != idx) {
            unite(parent, rank, idx, minIdx);
        }
    }
}

__global__ void propagateLabels(int *labels, int *parent, int *rank, int *minima, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int x = idx % width;
    int y = idx / width;
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        int minIdx = minima[idx];
        int minLabel = labels[minIdx];
        labels[idx] = minLabel;
        if (minIdx != idx) {
            unite(parent, rank, idx, minIdx);
        }
    }
}

__global__ void finalizeLabels(int *labels, int *parent, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width * height) {
        labels[idx] = find(parent, labels[idx]);
    }
}

void checkCudaError(hipError_t err, const char *msg) {
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << msg << " (" << hipGetErrorString(err) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    int width = 1024;
    int height = 1024;
    int numPixels = width * height;

    thrust::host_vector<int> h_image(numPixels);
    thrust::host_vector<int> h_labels(numPixels);
    thrust::host_vector<int> h_minima(numPixels);

    // Generate synthetic image data
    for (int i = 0; i < numPixels; ++i) {
        h_image[i] = rand() % 256;
    }

    thrust::device_vector<int> d_image = h_image;
    thrust::device_vector<int> d_labels(numPixels);
    thrust::device_vector<int> d_minima(numPixels);
    thrust::device_vector<int> d_parent(numPixels);
    thrust::device_vector<int> d_rank(numPixels);

    int *d_image_ptr = thrust::raw_pointer_cast(d_image.data());
    int *d_labels_ptr = thrust::raw_pointer_cast(d_labels.data());
    int *d_minima_ptr = thrust::raw_pointer_cast(d_minima.data());
    int *d_parent_ptr = thrust::raw_pointer_cast(d_parent.data());
    int *d_rank_ptr = thrust::raw_pointer_cast(d_rank.data());

    int numBlocks = (numPixels + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Initialize labels, parent, and rank
    initializeLabels<<<numBlocks, BLOCK_SIZE>>>(d_labels_ptr, d_parent_ptr, d_rank_ptr, width, height);
    checkCudaError(hipGetLastError(), "initializeLabels");

    // Find minima
    findMinima<<<numBlocks, BLOCK_SIZE>>>(d_labels_ptr, d_parent_ptr, d_rank_ptr, d_minima_ptr, d_image_ptr, width, height);
    checkCudaError(hipGetLastError(), "findMinima");

    // Propagate labels
    propagateLabels<<<numBlocks, BLOCK_SIZE>>>(d_labels_ptr, d_parent_ptr, d_rank_ptr, d_minima_ptr, width, height);
    checkCudaError(hipGetLastError(), "propagateLabels");

    // Finalize labels
    finalizeLabels<<<numBlocks, BLOCK_SIZE>>>(d_labels_ptr, d_parent_ptr, width, height);
    checkCudaError(hipGetLastError(), "finalizeLabels");

    // Copy results back to host
    thrust::copy(d_labels.begin(), d_labels.end(), h_labels.begin());

    // Output first 10 labels for verification
    for (int i = 0; i < 10; ++i) {
        std::cout << "Label at index " << i << ": " << h_labels[i] << std::endl;
    }

    return 0;
}