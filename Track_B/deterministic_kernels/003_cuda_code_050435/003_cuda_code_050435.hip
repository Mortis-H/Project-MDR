#include <iostream>
#include <hip/hip_runtime.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/copy.h>

// Define constants
#define BLOCK_SIZE 16
#define MAX_LEVELS 256

// Kernel to initialize markers
__global__ void initializeMarkers(unsigned char *image, int *markers, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        if (image[idx] == 0) {
            markers[idx] = idx + 1; // Background
        } else {
            markers[idx] = 0; // Foreground
        }
    }
}

// Kernel to propagate markers
__global__ void propagateMarkers(int *markers, int width, int height, int level) {
    extern __shared__ int s_markers[];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    if (x >= width || y >= height) return;

    // Load data into shared memory
    int bx = threadIdx.x + blockDim.x * blockIdx.x;
    int by = threadIdx.y + blockDim.y * blockIdx.y;
    if (bx < width && by < height) {
        s_markers[threadIdx.y * blockDim.x + threadIdx.x] = markers[by * width + bx];
    } else {
        s_markers[threadIdx.y * blockDim.x + threadIdx.x] = 0;
    }
    __syncthreads();

    // Propagation logic
    if (markers[idx] == 0) {
        int min_marker = 0;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) continue;
                int nx = x + dx;
                int ny = y + dy;
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    int nidx = ny * width + nx;
                    int neighbor = markers[nidx];
                    if (neighbor != 0 && (min_marker == 0 || neighbor < min_marker)) {
                        min_marker = neighbor;
                    }
                }
            }
        }
        if (min_marker != 0) {
            markers[idx] = min_marker;
        }
    }
}

// Kernel to apply watershed transform
__global__ void watershedTransform(unsigned char *image, int *markers, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        if (markers[idx] == 0) {
            markers[idx] = idx + 1; // New region
        }
    }
}

// Main function to demonstrate the usage of the watershed segmentation
int main() {
    int width = 512;
    int height = 512;
    size_t imageSize = width * height * sizeof(unsigned char);
    size_t markersSize = width * height * sizeof(int);

    // Allocate host memory
    thrust::host_vector<unsigned char> h_image(imageSize);
    thrust::host_vector<int> h_markers(markersSize);

    // Initialize image with some data (simple example: a cross)
    thrust::fill(h_image.begin(), h_image.end(), 0);
    int midX = width / 2;
    int midY = height / 2;
    for (int x = 0; x < width; ++x) {
        h_image[midY * width + x] = 255;
    }
    for (int y = 0; y < height; ++y) {
        h_image[y * width + midX] = 255;
    }

    // Allocate device memory
    unsigned char *d_image;
    int *d_markers;
    hipMalloc(&d_image, imageSize);
    hipMalloc(&d_markers, markersSize);

    // Copy data to device
    hipMemcpy(d_image, h_image.data(), imageSize, hipMemcpyHostToDevice);

    // Initialize markers
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    initializeMarkers<<<gridSize, blockSize>>>(d_image, d_markers, width, height);
    hipDeviceSynchronize();

    // Propagate markers
    for (int level = 1; level < MAX_LEVELS; ++level) {
        propagateMarkers<<<gridSize, blockSize, blockSize.x * blockSize.y * sizeof(int)>>>(d_markers, width, height, level);
        hipDeviceSynchronize();
    }

    // Apply watershed transform
    watershedTransform<<<gridSize, blockSize>>>(d_image, d_markers, width, height);
    hipDeviceSynchronize();

    // Copy results back to host
    hipMemcpy(h_markers.data(), d_markers, markersSize, hipMemcpyDeviceToHost);

    // Output results (simple: print the first 100 markers)
    for (int i = 0; i < 100; ++i) {
        std::cout << h_markers[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    hipFree(d_image);
    hipFree(d_markers);

    return 0;
}