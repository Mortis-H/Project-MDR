#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

namespace cg = cooperative_groups;

// Kernel to process a 4K frame (3840x2160) in parallel
__global__ void processFrameKernel(unsigned char* d_frame, int width, int height) {
    // Calculate the row and column index for each thread
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is within the bounds of the frame
    if (col < width && row < height) {
        // Calculate the index in the 1D array
        int index = row * width + col;

        // Perform some processing on the pixel, for example, grayscale conversion
        // Assuming the frame is in RGB format, we convert to grayscale
        int r = d_frame[index * 3];
        int g = d_frame[index * 3 + 1];
        int b = d_frame[index * 3 + 2];

        // Grayscale formula: Y = 0.299R + 0.587G + 0.114B
        unsigned char gray = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);

        // Store the grayscale value in the same array (for simplicity)
        d_frame[index * 3] = gray;
        d_frame[index * 3 + 1] = gray;
        d_frame[index * 3 + 2] = gray;

        // Use cooperative groups for better synchronization
        cg::thread_block cta = cg::this_thread_block();
        cg::sync(cta);
    }
}

int main() {
    // Define the dimensions of a 4K frame
    int width = 3840;
    int height = 2160;
    int frameSize = width * height * 3 * sizeof(unsigned char);

    // Allocate memory for the frame on the host
    unsigned char* h_frame = new unsigned char[frameSize];

    // Initialize the frame with some values (random for demonstration)
    for (int i = 0; i < frameSize; ++i) {
        h_frame[i] = static_cast<unsigned char>(rand() % 256);
    }

    // Allocate memory for the frame on the device
    unsigned char* d_frame;
    hipError_t err = hipMalloc((void**)&d_frame, frameSize);
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory: " << hipGetErrorString(err) << std::endl;
        delete[] h_frame;
        return -1;
    }

    // Copy the frame from the host to the device
    err = hipMemcpy(d_frame, h_frame, frameSize, hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy data to device: " << hipGetErrorString(err) << std::endl;
        hipFree(d_frame);
        delete[] h_frame;
        return -1;
    }

    // Define the block and grid dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    processFrameKernel<<<gridSize, blockSize>>>(d_frame, width, height);

    // Check for errors in kernel launch
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel launch failed: " << hipGetErrorString(err) << std::endl;
        hipFree(d_frame);
        delete[] h_frame;
        return -1;
    }

    // Copy the processed frame back to the host
    err = hipMemcpy(h_frame, d_frame, frameSize, hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy data from device: " << hipGetErrorString(err) << std::endl;
        hipFree(d_frame);
        delete[] h_frame;
        return -1;
    }

    // Free the device memory
    hipFree(d_frame);

    // Free the host memory
    delete[] h_frame;

    std::cout << "Frame processed successfully!" << std::endl;
    return 0;
}