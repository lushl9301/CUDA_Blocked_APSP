#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <iostream>
#include <vector>
#include <cstdio>
#include <cuda_fp16.h>
#include <assert.h>
#include "fp16_conversion.h"

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include "APSP.cuh"

#define BLKSZ_1 32
#define BLKSZ_2 32
#define BLKSZ_3 32
#define ROUND 6

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}



__global__ void floyd_kernel_1(int k, half *devMat, int N)
{
	__shared__ half d_d[BLKSZ_1][BLKSZ_1];
	int base = k * BLKSZ_1;
	int i = threadIdx.y;
	int j = threadIdx.x;
	int d_i = i + base;
	int d_j = j + base;
	base = d_i * N + d_j;
	d_d[i][j] = devMat[base];
	__syncthreads();

	half newD;
	for (int t = 0; t < BLKSZ_1; t++) {
		newD = __hadd(d_d[i][t], d_d[t][j]);
		if (__hlt(newD, d_d[i][j]))
			d_d[i][j] = newD;
	}
	devMat[base] = d_d[i][j];
}

__global__ void floyd_kernel_2(int k, half *devMat, int N)
{
	if (blockIdx.x == k) return;

	__shared__ half d_d[BLKSZ_2][BLKSZ_2], d_c[BLKSZ_2][BLKSZ_2];
	int base = k * BLKSZ_2;
	int i = threadIdx.y;
	int j = threadIdx.x;

	int d_i = i + base;
	int d_j = j + base;
	base = d_i * N + d_j;
	d_d[i][j] = devMat[base]; // (k,k) matrix

	if (blockIdx.y == 0) {
		d_j = BLKSZ_2 * blockIdx.x + threadIdx.x;
	} else {
		d_i = BLKSZ_2 * blockIdx.x + threadIdx.y;
	}

	int current_base = d_i * N + d_j;
	d_c[i][j] = devMat[current_base]; // updating matrix
	__syncthreads();

	half newD;

	if (blockIdx.y == 0) {
		for (int t = 0; t < BLKSZ_2; t++) {
			newD = __hadd(d_d[i][t], d_c[t][j]);
			if (__hlt(newD, d_c[i][j]))
				d_c[i][j] = newD;
		}
	} else {
		for (int t = 0; t < BLKSZ_2; t++) {
			newD = __hadd(d_c[i][t], d_d[t][j]);
			if (__hlt(newD, d_c[i][j]))
				d_c[i][j] = newD;
		}
	}

	devMat[current_base] = d_c[i][j];
}


__global__ void floyd_kernel_3(int k, half *devMat, int N)
{
	if (blockIdx.x == k || blockIdx.y == k) return;

	__shared__ half d_c[BLKSZ_3][BLKSZ_3], d_r[BLKSZ_3][BLKSZ_3];
	int base = k * BLKSZ_3;
	int d_i = blockDim.y * blockIdx.y + threadIdx.y;
	int d_j = blockDim.x * blockIdx.x + threadIdx.x;
	int i = threadIdx.y;
	int j = threadIdx.x;
	int col_base = (base + i) * N + d_j;
	int row_base = d_i * N + base + j;
	base = d_i * N + d_j;

	d_r[i][j] = devMat[col_base];
	d_c[i][j] = devMat[row_base];
	half oldD = devMat[base];
	__syncthreads();
	half newD;
	for (int t = 0; t < BLKSZ_3; t++) {
		newD = __hadd(d_c[i][t], d_r[t][j]);
		if (__hlt(newD, oldD))
			oldD = newD;
	}
	devMat[base] = oldD;
}


int floyd(half* devMat, int N)
{
	int N_blk = N / BLKSZ_1;

	dim3 blockSize1(BLKSZ_1, BLKSZ_1);

	dim3 gridSizeP2(N / BLKSZ_2, 2);
	dim3 blockSize2(BLKSZ_2, BLKSZ_2);

	dim3 gridSizeP3(N / BLKSZ_3, N / BLKSZ_3);
	dim3 blockSize3(BLKSZ_3, BLKSZ_3);
 	for (int k = 0; k < N_blk; k++) {
		cudaProfilerStart();
		floyd_kernel_1 <<<1, blockSize1>>> (k, devMat, N);
		floyd_kernel_2 <<<gridSizeP2, blockSize2>>> (k, devMat, N);
		cudaProfilerStop();
		floyd_kernel_3 <<<gridSizeP3, blockSize3>>> (k, devMat, N);
	}

	gpuErrchk(cudaPeekAtLastError());
	return 0;
}


int main(int argc, char* argv[])
{
	int N;
	int flag = 0;
	int *mat, *ref;
	half *result;
	struct timespec start, end;
	double diff1, diff2;

	if (argc == 1) {
		printf("USAGE N [1]; N graph size, N = 2^k; 1 means doing validation\n");
		exit(-1);
	}
	if (argc == 2) {
		if (!sscanf(argv[1], "%d", &N)) {
			printf("USAGE N [1]; N graph size, N = 2^k; 1 means doing validation\n");
			exit(-1);
		}
	}
	if (argc == 3) {
		if (!sscanf(argv[1], "%d", &N)) {
			printf("USAGE N [1]; N graph size, N = 2^k; 1 means doing validation\n");
			exit(-1);
		}
		if (!sscanf(argv[2], "%d", &flag)) {
			printf("USAGE N [1]; N graph size, N = 2^k; 1 means doing validation\n");
			exit(-1);
		}
	}
	int ttt;
	cudaGetDeviceCount(&ttt);
	printf("device number = %d\n", ttt);

	mat = (int *)malloc(N * N * sizeof(int));
	ref = (int *)malloc(sizeof(int) * N * N);
	GenMatrix(mat, N);

	if (flag) {
		memcpy(ref, mat, sizeof(int) * N * N);
		clock_gettime(CLOCK_MONOTONIC, &start);
		ST_APSP(ref, N);
		clock_gettime(CLOCK_MONOTONIC, &end);
		diff1 = 1000000 * (end.tv_sec-start.tv_sec) + (end.tv_nsec-start.tv_nsec)/1000;
	}

	cudaMallocManaged(&result, N * N * sizeof(half));
//	cudaMemcpy(result, mat, sizeof(int) * N * N, cudaMemcpyHostToDevice);
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			result[i * N + j] = approx_float_to_half((float)(mat[i * N + j]));
		}
	}

	diff2 = 0;
	floyd(result, N);
	floyd(result, N);
	for (int i = 0; i < ROUND; i++) {
		clock_gettime(CLOCK_MONOTONIC, &start);
		floyd(result, N);
		cudaDeviceSynchronize();
		clock_gettime(CLOCK_MONOTONIC, &end);
		diff2 = 1000000 * (end.tv_sec-start.tv_sec) + (end.tv_nsec-start.tv_nsec)/1000;
		printf("%d Problem SIZE ------ CUDA_APSP elasped time =%.2lfus\n", N, diff2);
	}
	if (flag) {
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				mat[i* N + j] = (int)half_to_float(result[i* N + j]);
			}
		}
		if (CmpArray(mat, ref, N * N))
			printf("Your result is correct.\n");
		else
			printf("Your result is wrong.\n");

		printf("ST_APSP elasped time   = %.2lf us\n", diff1);
		printf("============================================\n");
		printf("Speedup = %.2lf\n", diff1 / diff2);
	}
}
