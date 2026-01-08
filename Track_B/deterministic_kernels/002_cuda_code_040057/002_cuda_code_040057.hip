#include <iostream>
#include <hip/hip_runtime.h>


#define N 1024

__device__ void heapifyUp(int* d_heap, int idx) {
    int parentIdx = (idx - 1) / 2;
    while (idx > 0 && d_heap[parentIdx] < d_heap[idx]) {
        int temp = d_heap[parentIdx];
        d_heap[parentIdx] = d_heap[idx];
        d_heap[idx] = temp;
        idx = parentIdx;
        parentIdx = (idx - 1) / 2;
    }
}

__device__ void heapifyDown(int* d_heap, int size, int idx) {
    int largest = idx;
    int left = 2 * idx + 1;
    int right = 2 * idx + 2;

    if (left < size && d_heap[left] > d_heap[largest])
        largest = left;

    if (right < size && d_heap[right] > d_heap[largest])
        largest = right;

    if (largest != idx) {
        int temp = d_heap[idx];
        d_heap[idx] = d_heap[largest];
        d_heap[largest] = temp;
        heapifyDown(d_heap, size, largest);
    }
}

__global__ void buildHeapKernel(int* d_heap, int n) {
    extern __shared__ int s_heap[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    if (idx < n) {
        s_heap[tid] = d_heap[idx];
    } else {
        s_heap[tid] = INT_MIN; // Ensure out-of-bound indices are valid
    }
    __syncthreads();

    // Build heap in shared memory
    for (int s = blockDim.x / 2 - 1; s >= 0; --s) {
        if (tid <= s) {
            heapifyUp(s_heap, tid);
        }
        __syncthreads();
    }

    // Write back to global memory
    if (idx < n) {
        d_heap[idx] = s_heap[tid];
    }
}

__global__ void extractMaxKernel(int* d_heap, int n) {
    extern __shared__ int s_heap[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    if (idx < n) {
        s_heap[tid] = d_heap[idx];
    } else {
        s_heap[tid] = INT_MIN; // Ensure out-of-bound indices are valid
    }
    __syncthreads();

    // Extract max element and heapify down
    if (tid == 0) {
        int maxElement = s_heap[0];
        s_heap[0] = s_heap[n - 1];
        heapifyDown(s_heap, n - 1, 0);
        s_heap[n - 1] = maxElement; // Store max element at the end
    }
    __syncthreads();

    // Write back to global memory
    if (idx < n) {
        d_heap[idx] = s_heap[tid];
    }
}

int main() {
    int h_heap[N];
    for (int i = 0; i < N; ++i) {
        h_heap[i] = rand() % 1000; // Initialize heap with random values
    }

    int* d_heap;
    hipError_t err = hipMalloc((void**)&d_heap, N * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    err = hipMemcpy(d_heap, h_heap, N * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy data to device: " << hipGetErrorString(err) << std::endl;
        hipFree(d_heap);
        return -1;
    }

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // Build heap
    buildHeapKernel<<<numBlocks, blockSize, blockSize * sizeof(int)>>>(d_heap, N);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel launch failed: " << hipGetErrorString(err) << std::endl;
        hipFree(d_heap);
        return -1;
    }

    // Extract max element
    extractMaxKernel<<<1, blockSize, blockSize * sizeof(int)>>>(d_heap, N);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel launch failed: " << hipGetErrorString(err) << std::endl;
        hipFree(d_heap);
        return -1;
    }

    err = hipMemcpy(h_heap, d_heap, N * sizeof(int), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy data from device: " << hipGetErrorString(err) << std::endl;
        hipFree(d_heap);
        return -1;
    }

    // Print the max element
    std::cout << "Max element extracted: " << h_heap[N - 1] << std::endl;

    hipFree(d_heap);
    return 0;
}