#include <iostream>
#include <hip/hip_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>

#define N 512

__global__ void insert_into_heap(int *heap, int *values, int *indices, int num_values) {
    extern __shared__ int s_heap[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    if (idx < num_values) {
        s_heap[tid] = values[idx];
        indices[idx] = idx;
    } else {
        s_heap[tid] = INT_MIN;
    }
    __syncthreads();

    // Insert into heap
    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        int swap = (tid % (2 * offset) >= offset) && (s_heap[tid] > s_heap[tid - offset]);
        int tmp = s_heap[tid];
        s_heap[tid] = swap ? s_heap[tid - offset] : s_heap[tid];
        s_heap[tid - offset] = swap ? tmp : s_heap[tid - offset];
        __syncthreads();
    }
    __syncthreads();

    // Write back to global memory
    if (idx < num_values) {
        heap[tid] = s_heap[tid];
    }
}

__global__ void extract_max_from_heap(int *heap, int *indices, int *max_value, int *max_index) {
    extern __shared__ int s_heap[];
    int tid = threadIdx.x;

    // Load data into shared memory
    s_heap[tid] = heap[tid];
    __syncthreads();

    // Extract max element
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        int swap = (tid < offset) && (s_heap[tid] < s_heap[tid + offset]);
        int tmp = s_heap[tid];
        s_heap[tid] = swap ? s_heap[tid + offset] : s_heap[tid];
        s_heap[tid + offset] = swap ? tmp : s_heap[tid + offset];
        __syncthreads();
    }
    __syncthreads();

    // Write max element to global memory
    if (tid == 0) {
        *max_value = s_heap[0];
        *max_index = indices[0];
    }
}

int main() {
    thrust::host_vector<int> h_values(N);
    thrust::sequence(h_values.begin(), h_values.end()); // Fill with 0 to N-1
    thrust::host_vector<int> h_indices(N);
    thrust::device_vector<int> d_values = h_values;
    thrust::device_vector<int> d_indices = h_indices;
    thrust::device_vector<int> d_heap(N);
    int *d_max_value, *d_max_index;
    hipMalloc(&d_max_value, sizeof(int));
    hipMalloc(&d_max_index, sizeof(int));

    int *d_heap_ptr = thrust::raw_pointer_cast(d_heap.data());
    int *d_values_ptr = thrust::raw_pointer_cast(d_values.data());
    int *d_indices_ptr = thrust::raw_pointer_cast(d_indices.data());

    // Insert values into heap
    insert_into_heap<<<1, N, N * sizeof(int)>>>(d_heap_ptr, d_values_ptr, d_indices_ptr, N);
    hipDeviceSynchronize();

    // Extract max from heap
    extract_max_from_heap<<<1, N, N * sizeof(int)>>>(d_heap_ptr, d_indices_ptr, d_max_value, d_max_index);
    hipDeviceSynchronize();

    // Copy results back to host
    int h_max_value, h_max_index;
    hipMemcpy(&h_max_value, d_max_value, sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(&h_max_index, d_max_index, sizeof(int), hipMemcpyDeviceToHost);

    // Output the results
    std::cout << "Max value: " << h_max_value << " at index: " << h_max_index << std::endl;

    // Free device memory
    hipFree(d_max_value);
    hipFree(d_max_index);

    return 0;
}