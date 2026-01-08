#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>

#define BLOCK_SIZE 16

// Kernel to initialize the markers for the watershed algorithm
__global__ void initializeMarkers(int* markers, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    if (x < width && y < height) {
        // Simple initialization: set borders as markers
        if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
            markers[idx] = 1;
        } else {
            markers[idx] = 0;
        }
    }
}

// Kernel to perform a single pass of the watershed transformation
__global__ void watershedPass(int* image, int* markers, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        int currentMarker = markers[idx];
        int neighborMarker = 0;

        // Check neighbors for markers
        if (markers[idx - 1] != 0) neighborMarker = markers[idx - 1];
        if (markers[idx + 1] != 0) neighborMarker = markers[idx + 1];
        if (markers[idx - width] != 0) neighborMarker = markers[idx - width];
        if (markers[idx + width] != 0) neighborMarker = markers[idx + width];

        // If a neighbor has a marker and current pixel does not, assign the marker
        if (currentMarker == 0 && neighborMarker != 0) {
            markers[idx] = neighborMarker;
        }
    }
}

int main() {
    int width = 256;
    int height = 256;
    size_t imageSize = width * height * sizeof(int);

    // Allocate host memory
    thrust::host_vector<int> h_image(imageSize / sizeof(int));
    thrust::host_vector<int> h_markers(imageSize / sizeof(int));

    // Initialize image and markers on host
    thrust::sequence(h_image.begin(), h_image.end());
    thrust::fill(h_markers.begin(), h_markers.end(), 0);

    // Allocate device memory
    int* d_image;
    int* d_markers;
    hipError_t err = hipMalloc((void**)&d_image, imageSize);
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for image: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc((void**)&d_markers, imageSize);
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for markers: " << hipGetErrorString(err) << std::endl;
        hipFree(d_image);
        return -1;
    }

    // Asynchronous memory copy from host to device
    hipMemcpyAsync(d_image, h_image.data(), imageSize, hipMemcpyHostToDevice, 0);
    hipMemcpyAsync(d_markers, h_markers.data(), imageSize, hipMemcpyHostToDevice, 0);

    // Initialize markers
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    initializeMarkers<<<gridSize, blockSize>>>(d_markers, width, height);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel launch failed for initializeMarkers: " << hipGetErrorString(err) << std::endl;
        hipFree(d_image);
        hipFree(d_markers);
        return -1;
    }

    // Perform watershed transformation
    for (int i = 0; i < 10; ++i) { // Perform multiple passes
        watershedPass<<<gridSize, blockSize>>>(d_image, d_markers, width, height);
        err = hipGetLastError();
        if (err != hipSuccess) {
            std::cerr << "Kernel launch failed for watershedPass: " << hipGetErrorString(err) << std::endl;
            hipFree(d_image);
            hipFree(d_markers);
            return -1;
        }
    }

    // Asynchronous memory copy from device to host
    hipMemcpyAsync(h_markers.data(), d_markers, imageSize, hipMemcpyDeviceToHost, 0);

    // Synchronize to ensure all operations are complete
    hipDeviceSynchronize();

    // Free device memory
    hipFree(d_image);
    hipFree(d_markers);

    // Output the result (for demonstration purposes, print the first 10 markers)
    for (int i = 0; i < 10; ++i) {
        std::cout << "Marker[" << i << "] = " << h_markers[i] << std::endl;
    }

    return 0;
}