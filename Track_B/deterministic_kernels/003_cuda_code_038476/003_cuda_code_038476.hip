#include <iostream>
#include <hip/hip_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/adjacent_difference.h>
#include <thrust/scan.h>
#include <thrust/extrema.h>

#define BLOCK_SIZE 256

__device__ int find(int *parent, int x) {
    while (parent[x] != x) {
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

__global__ void label_pixels(int *image, int *markers, int *parent, int *rank, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int x = idx % width;
    int y = idx / width;
    if (x >= width || y >= height) return;

    int label = markers[idx];
    if (label != 0) {
        parent[idx] = idx;
        rank[idx] = 0;
    } else {
        parent[idx] = -1;
    }
}

__global__ void propagate_labels(int *image, int *parent, int *rank, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int x = idx % width;
    int y = idx / width;
    if (x >= width || y >= height) return;

    int offsets[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    int min_label = -1;
    for (int i = 0; i < 4; i++) {
        int nx = x + offsets[i][0];
        int ny = y + offsets[i][1];
        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
            int neighbor_idx = ny * width + nx;
            if (parent[neighbor_idx] != -1) {
                int neighbor_label = find(parent, neighbor_idx);
                if (min_label == -1 || neighbor_label < min_label) {
                    min_label = neighbor_label;
                }
            }
        }
    }
    if (min_label != -1 && parent[idx] != -1) {
        unite(parent, rank, idx, min_label);
    }
}

__global__ void assign_labels(int *image, int *parent, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int x = idx % width;
    int y = idx / width;
    if (x >= width || y >= height) return;

    if (parent[idx] != -1) {
        image[idx] = find(parent, idx);
    }
}

int main() {
    int width = 1024;
    int height = 1024;
    int num_pixels = width * height;

    // Initialize host data
    thrust::host_vector<int> h_image(num_pixels, 0);
    thrust::host_vector<int> h_markers(num_pixels, 0);

    // Example: Place markers
    h_markers[0] = 1;
    h_markers[num_pixels - 1] = 2;

    // Allocate device memory
    int *d_image, *d_markers, *d_parent, *d_rank;
    hipMalloc(&d_image, num_pixels * sizeof(int));
    hipMalloc(&d_markers, num_pixels * sizeof(int));
    hipMalloc(&d_parent, num_pixels * sizeof(int));
    hipMalloc(&d_rank, num_pixels * sizeof(int));

    // Copy data to device
    hipMemcpy(d_image, h_image.data(), num_pixels * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_markers, h_markers.data(), num_pixels * sizeof(int), hipMemcpyHostToDevice);

    // Launch kernels
    int num_blocks = (num_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE;

    label_pixels<<<num_blocks, BLOCK_SIZE>>>(d_image, d_markers, d_parent, d_rank, width, height);
    hipDeviceSynchronize();

    for (int iter = 0; iter < 5; iter++) { // Assuming a fixed number of iterations for simplicity
        propagate_labels<<<num_blocks, BLOCK_SIZE>>>(d_image, d_parent, d_rank, width, height);
        hipDeviceSynchronize();
    }

    assign_labels<<<num_blocks, BLOCK_SIZE>>>(d_image, d_parent, width, height);
    hipDeviceSynchronize();

    // Copy results back to host
    thrust::device_vector<int> d_image_thrust(d_image, d_image + num_pixels);
    thrust::copy(d_image_thrust.begin(), d_image_thrust.end(), h_image.begin());

    // Free device memory
    hipFree(d_image);
    hipFree(d_markers);
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