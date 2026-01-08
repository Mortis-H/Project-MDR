#include <iostream>
#include <hip/hip_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

#define NUM_ELEMENTS 16384
#define BLOCK_SIZE 256

__device__ int d_heap[NUM_ELEMENTS];

__device__ __inline__ void swap(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

__device__ __inline__ void heapify_up(int idx) {
    while (idx > 0) {
        int parent = (idx - 1) / 2;
        if (d_heap[idx] > d_heap[parent]) {
            swap(&d_heap[idx], &d_heap[parent]);
            idx = parent;
        } else {
            break;
        }
    }
}

__device__ __inline__ void heapify_down(int idx, int n) {
    while (true) {
        int left = 2 * idx + 1;
        int right = 2 * idx + 2;
        int largest = idx;

        if (left < n && d_heap[left] > d_heap[largest]) {
            largest = left;
        }
        if (right < n && d_heap[right] > d_heap[largest]) {
            largest = right;
        }
        if (largest != idx) {
            swap(&d_heap[idx], &d_heap[largest]);
            idx = largest;
        } else {
            break;
        }
    }
}

__global__ void build_heap_kernel(int* data, int n) {
    extern __shared__ int sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx < n) {
        sdata[tid] = data[idx];
    } else {
        sdata[tid] = -1; // Initialize with a value that won't affect the max heap
    }
    __syncthreads();

    // Build the heap in shared memory
    for (int s = blockDim.x / 2 - 1; s >= 0; --s) {
        if (tid <= s) {
            int left = 2 * tid + 1;
            int right = 2 * tid + 2;
            int largest = tid;

            if (left < blockDim.x && sdata[left] > sdata[largest]) {
                largest = left;
            }
            if (right < blockDim.x && sdata[right] > sdata[largest]) {
                largest = right;
            }
            if (largest != tid) {
                swap(&sdata[tid], &sdata[largest]);
            }
        }
        __syncthreads();
    }

    // Write the heap back to global memory
    if (idx < n) {
        d_heap[idx] = sdata[tid];
    }
}

__global__ void insert_element_kernel(int element, int n) {
    int tid = threadIdx.x;

    if (tid == 0) {
        d_heap[n] = element;
        heapify_up(n);
    }
}

__global__ void extract_max_kernel(int* result, int n) {
    int tid = threadIdx.x;

    if (tid == 0) {
        *result = d_heap[0];
        d_heap[0] = d_heap[n - 1];
        heapify_down(0, n - 1);
    }
}

int main() {
    thrust::host_vector<int> h_data(NUM_ELEMENTS);
    for (int i = 0; i < NUM_ELEMENTS; ++i) {
        h_data[i] = rand() % 100000;
    }

    thrust::device_vector<int> d_data = h_data;
    int* d_data_ptr = thrust::raw_pointer_cast(d_data.data());

    int* d_result;
    hipMalloc(&d_result, sizeof(int));

    int num_blocks = (NUM_ELEMENTS + BLOCK_SIZE - 1) / BLOCK_SIZE;

    build_heap_kernel<<<num_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(d_data_ptr, NUM_ELEMENTS);
    hipDeviceSynchronize();

    int new_element = 100001;
    insert_element_kernel<<<1, BLOCK_SIZE>>>(new_element, NUM_ELEMENTS + 1);
    hipDeviceSynchronize();

    extract_max_kernel<<<1, BLOCK_SIZE>>>(d_result, NUM_ELEMENTS + 1);
    hipDeviceSynchronize();

    int h_result;
    hipMemcpy(&h_result, d_result, sizeof(int), hipMemcpyDeviceToHost);

    std::cout << "Max element extracted: " << h_result << std::endl;

    hipFree(d_result);

    return 0;
}