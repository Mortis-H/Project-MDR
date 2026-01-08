#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <hip/hip_runtime.h>

using namespace std;

__global__ void block_scan(int *output, int *input, int *sums, int n)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int blockOff = bid * n;

    extern __shared__ int buffer[];

    buffer[2 * tid] = input[blockOff + (2 * tid)];
    buffer[2 * tid + 1] = input[blockOff + (2 * tid) + 1];

    int offset = 1;
    for (int d = n >> 1; d > 0; d >>= 1)
    {
        __syncthreads();
        if (tid < d)
        {
            int a = offset * (2 * tid + 1) - 1;
            int b = offset * (2 * tid + 2) - 1;
            buffer[b] += buffer[a];
        }

        offset *= 2;
    }

    __syncthreads();

    if (tid == 0)
    {
        sums[bid] = buffer[n - 1];
        buffer[n - 1] = 0;
    }

    for (int d = 1; d < n; d *= 2)
    {
        offset >>= 1;
        __syncthreads();
        if (tid < d)
        {
            int a = offset * (2 * tid + 1) - 1;
            int b = offset * (2 * tid + 2) - 1;
            int t = buffer[a];
            buffer[a] = buffer[b];
            buffer[b] += t;
        }
    }

    __syncthreads();

    output[blockOff + (2 * tid)] = buffer[2 * tid];
    output[blockOff + (2 * tid) + 1] = buffer[2 * tid + 1];
}

__global__ void add(int *output, int length, int *n)
{
    int blockId = blockIdx.x;
    int tid = threadIdx.x;
    int blockOffset = blockId * length;

    output[blockOffset + tid] += n[blockId];
}

__global__ void markBit(int *input, int *predicates, int bit, int length)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= length)
    {
        return;
    }

    int mask = 1 << bit;
    predicates[idx] = input[idx] & mask ? 0 : 1;
}

__global__ void compact(int *input, int *output, int *f, int *t, int *predicate, int length)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= length)
    {
        return;
    }

    t[idx] = idx - f[idx] + f[length];

    if (predicate[idx] == 0)
    {
        int address = t[idx];
        output[address] = input[idx];
    }
    else
    {
        int address = f[idx];
        output[address] = input[idx];
    }
}

int main()
{
    vector<int> data;
    ifstream infile;
    infile.open("inp.txt");

    if (infile.is_open())
    {
        while (infile.good())
        {
            char cNum[10];
            infile.getline(cNum, 256, ',');
            int num = atoi(cNum);
            data.push_back(num);
            // cout << num << " ";
        }
        infile.close();
    }
    else
    {
        cout << "Error opening file";
    }

    int size = data.size();
    int size1 = size * sizeof(int);

    int *d_f;
    int *d_true;
    int *d_input;
    int *d_output;
    int *d_predicates;
    int *d_result;
    int *d_dummy_blocks_sums;
    int *d_sums;
    int *d_inc;

    int *output = (int *)malloc(size1);

    int blocks = size / 1024;
    if (size % 1024 != 0)
    {
        blocks += 1;
    }

    hipMalloc((void **)&d_f, size1);
    hipMalloc((void **)&d_true, size1);
    hipMalloc((void **)&d_input, size1);
    hipMalloc((void **)&d_output, size1);
    hipMalloc((void **)&d_predicates, size1);
    hipMalloc((void **)&d_result, size1);
    hipMalloc((void **)&d_sums, blocks * sizeof(int));
    hipMalloc((void **)&d_inc, blocks * sizeof(int));
    hipMalloc((void **)&d_dummy_blocks_sums, blocks * sizeof(int));

    hipMemcpy(d_output, data.data(), size1, hipMemcpyHostToDevice);
    hipMemcpy(d_input, data.data(), size1, hipMemcpyHostToDevice);
    hipDeviceSynchronize();

    const int sharedSize = 2 * 1024 * sizeof(int);

    for (int i = 0; i < 10; i++)
    {
        markBit<<<blocks, 1024>>>(d_input, d_predicates, i, size);
        block_scan<<<blocks, 512, sharedSize>>>(d_f, d_predicates, d_sums, 1024);
        block_scan<<<1, (blocks + 1) / 2, sharedSize>>>(d_inc, d_sums, d_dummy_blocks_sums, 1024);
        add<<<blocks, 1024>>>(d_f, 1024, d_inc);
        compact<<<blocks, 1024>>>(d_input, d_result, d_f, d_true, d_predicates, size);
        hipMemcpy(d_input, d_result, size1, hipMemcpyDeviceToDevice);
        hipDeviceSynchronize();
    }

    hipMemcpy(output, d_input, size1, hipMemcpyDeviceToHost);

    ofstream outfile;
    outfile.open("q4.txt");

    if (outfile.is_open())
    {

        for (int i = 0; i < size; i++)
        {
            outfile << output[i] << ", ";
        }

        outfile.close();
    }
    else
    {
        cout << "Error opening file";
    }

    hipFree(d_f);
    hipFree(d_true);
    hipFree(d_input);
    hipFree(d_sums);
    hipFree(d_inc);
    hipFree(d_predicates);
    hipFree(d_dummy_blocks_sums);

    free(output);
}