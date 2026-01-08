#include <iostream>
#include <hip/hip_runtime.h>


#define N 16384
#define BLOCK_SIZE 256

__global__ void initializeQueue(int *queue, int *indices, int *sizes, int value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        queue[idx] = value;
        indices[idx] = idx;
        sizes[idx] = 0;
    }
}

__global__ void insertIntoQueue(int *queue, int *indices, int *sizes, int value, int idx) {
    if (idx < N) {
        int pos = atomicAdd(&sizes[idx], 1);
        queue[idx * N + pos] = value;
        // Simple bubble up to maintain heap property
        while (pos > 0) {
            int parent = (pos - 1) / 2;
            if (queue[idx * N + parent] < queue[idx * N + pos]) {
                int temp = queue[idx * N + parent];
                queue[idx * N + parent] = queue[idx * N + pos];
                queue[idx * N + pos] = temp;
                pos = parent;
            } else {
                break;
            }
        }
    }
}

__global__ void extractMaxFromQueue(int *queue, int *indices, int *sizes, int *result, int idx) {
    if (idx < N) {
        int size = atomicSub(&sizes[idx], 1);
        if (size > 0) {
            result[idx] = queue[idx * N];
            queue[idx * N] = queue[idx * N + size - 1];
            // Simple bubble down to maintain heap property
            int pos = 0;
            while (true) {
                int left = 2 * pos + 1;
                int right = 2 * pos + 2;
                int largest = pos;
                if (left < size && queue[idx * N + left] > queue[idx * N + largest]) {
                    largest = left;
                }
                if (right < size && queue[idx * N + right] > queue[idx * N + largest]) {
                    largest = right;
                }
                if (largest != pos) {
                    int temp = queue[idx * N + pos];
                    queue[idx * N + pos] = queue[idx * N + largest];
                    queue[idx * N + largest] = temp;
                    pos = largest;
                } else {
                    break;
                }
            }
        } else {
            result[idx] = -1; // Indicate no elements left
        }
    }
}

int main() {
    int *d_queue, *d_indices, *d_sizes, *d_result;
    int h_queue[N * N], h_indices[N], h_sizes[N], h_result[N];

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_indices[i] = i;
        h_sizes[i] = 0;
        for (int j = 0; j < N; ++j) {
            h_queue[i * N + j] = 0;
        }
    }

    // Allocate device memory
    hipError_t err = hipMalloc((void **)&d_queue, N * N * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for queue" << std::endl;
        return -1;
    }
    err = hipMalloc((void **)&d_indices, N * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for indices" << std::endl;
        return -1;
    }
    err = hipMalloc((void **)&d_sizes, N * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for sizes" << std::endl;
        return -1;
    }
    err = hipMalloc((void **)&d_result, N * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for result" << std::endl;
        return -1;
    }

    // Copy data from host to device
    err = hipMemcpy(d_queue, h_queue, N * N * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy queue from host to device" << std::endl;
        return -1;
    }
    err = hipMemcpy(d_indices, h_indices, N * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy indices from host to device" << std::endl;
        return -1;
    }
    err = hipMemcpy(d_sizes, h_sizes, N * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy sizes from host to device" << std::endl;
        return -1;
    }

    // Initialize the queue
    initializeQueue<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_queue, d_indices, d_sizes, 0);
    hipDeviceSynchronize();

    // Insert values into the queue
    for (int i = 0; i < N; ++i) {
        insertIntoQueue<<<1, 1>>>(d_queue, d_indices, d_sizes, i, i);
        hipDeviceSynchronize();
    }

    // Extract maximum values from the queue
    extractMaxFromQueue<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_queue, d_indices, d_sizes, d_result, 0);
    hipDeviceSynchronize();

    // Copy result back to host
    err = hipMemcpy(h_result, d_result, N * sizeof(int), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy result from device to host" << std::endl;
        return -1;
    }

    // Free device memory
    hipFree(d_queue);
    hipFree(d_indices);
    hipFree(d_sizes);
    hipFree(d_result);

    // Print results
    std::cout << "Extracted maximum values:" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << h_result[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}