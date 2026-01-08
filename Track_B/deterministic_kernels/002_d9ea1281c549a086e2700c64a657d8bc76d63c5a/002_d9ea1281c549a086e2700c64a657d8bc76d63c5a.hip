#include<stdio.h>
#include<hip/hip_runtime.h>
#define N 8000

__global__ void fun(int *a, int aLen)
{
    unsigned int id = threadIdx.x;
    if(id < aLen)   a[id] = 0;
}

__global__ void add(int *a, int aLen)
{
    unsigned int id = threadIdx.x;
    if(id < aLen)   a[id] += id;
}

int main()
{
    int *da, i = 0;

    hipMalloc(&da, sizeof(int) * N);
    fun<<<1, N>>>(da, N);
    add<<<1, N>>>(da, N);

    int a[N];
    hipMemcpy(a, da, sizeof(int) * N, hipMemcpyDeviceToHost);

    for(i = 0; i < N; i++)
    {
        printf("%d ", a[i]);
    }

    return 0;
}