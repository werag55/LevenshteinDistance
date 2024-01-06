#include "CudaAlgorithm.cuh"
#include <iostream>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define CHECK_CUDA_ERROR(val) CudaAlg::check((val), #val, __FILE__, __LINE__)

//https://leimao.github.io/blog/Proper-CUDA-Error-Checking/
template <typename T>
void check(T err, const char* const func, const char* const file,
    const int line)
{
    if (err != cudaSuccess)
    {
        cerr << "CUDA Runtime Error at: " << file << ":" << line << endl;
        cerr << cudaGetErrorString(err) << " " << func << endl;
        exit(EXIT_FAILURE);
    }
}

__global__ void ComputeX(int* d_X, char* d_Q, int q, char* d_t, int n)
{
    int i = threadIdx.x;

    extern __shared__ char sharedQ[];
    sharedQ[i] = d_Q[i];
    __syncthreads();

    d_X[i * (n + 1) + 0] = 0;
    for (int j = 1; j <= n; j++)
    {
        if (d_t[j - 1] == sharedQ[i])
            d_X[i * (n + 1) + j] = j;
        else
            d_X[i * (n + 1) + j] = d_X[i * (n + 1) + j - 1];
    }
}

void printArray(int* array, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
            cout << array[i * n + j] << " ";
        cout << endl;
    }
    cout << endl <<endl;
}

__device__ int cudaMinimum(int a, int b, int c, char* t)
{
	int min = a;

	if (b < min)
		min = b;

	if (c < min)
		min = c;

	return min;
}

int CudaAlgorithm::LevenstheinDistance(string s, string t)
{
    m = s.size(); n = t.size();

    cudaMalloc((void**)&d_s, (m + 1) * sizeof(char));
    cudaMalloc((void**)&d_t, (n + 1) * sizeof(char));
    cudaMalloc((void**)&d_X, q * (n + 1) * sizeof(int));
    cudaMalloc((void**)&d_D, (m + 1) * (n + 1) * sizeof(int));

    char* h_s = new char[m + 1];
    strcpy(h_s, s.c_str());
    char* h_t = new char[n + 1];
    strcpy(h_t, t.c_str());
    cudaMemcpy(d_s, h_s, (m + 1) * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_t, h_t, (n + 1) * sizeof(char), cudaMemcpyHostToDevice);

    ComputeX<<<1,q>>>(d_X, d_Q, q, d_t, n);
    //int* h_X = new int[q * (n + 1)];
    //cudaMemcpy(h_X, d_X, q * (n + 1) * sizeof(int), cudaMemcpyDeviceToHost);

    //printArray(h_X, q, n + 1);



    return 0;
}

CudaAlgorithm::CudaAlgorithm()
{
    cudaMalloc((void**)&d_Q, (q + 1) * sizeof(char));
    cudaMemcpy(d_Q, h_Q, (q + 1) * sizeof(char), cudaMemcpyHostToDevice);
}