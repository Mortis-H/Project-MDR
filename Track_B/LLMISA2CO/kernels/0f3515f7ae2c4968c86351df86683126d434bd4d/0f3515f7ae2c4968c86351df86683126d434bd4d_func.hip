__global__ void uniqueIdxCalcUsingThreadIdx(int* arr) {
	unsigned int tidx{ threadIdx.x };
	printf("threadIdx : %d,   value : %d \n", tidx, arr[tidx]);
}