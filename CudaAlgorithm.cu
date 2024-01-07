#include "CudaAlgorithm.cuh"
#include <iostream>
#include <algorithm>
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

__device__ int cudaMinimum(int a, int b, int c, char* t)
{
    int min = a;
    *t = 's';

    if (b < min)
    {
        min = b;
        *t = 'd';
    }

    if (c < min)
    {
        min = c;
        *t = 'i';
    }

    return min;
}

__global__ void ComputeD(int* d_D, char* d_T, int* d_X, 
    char* d_s, int m, char* d_t, int n, int w)
{
    int j = threadIdx.x;
    if (j > n)
        return;

    int Dvar, Avar, Bvar, Cvar;
    char T;

    Dvar = j;
    d_D[j] = Dvar;
    d_T[j] = 'i';
    //__syncthreads();

    for (int i = 1; i <= m; i++)
    {
        __syncthreads();
        int shflVal = __shfl_up(Dvar, 1);
        if (j % w == 0 && j != 0)
            Avar = d_D[(i - 1) * (n + 1) + j - 1];
        else if (j != 0)
            Avar = shflVal;
        __syncthreads();

        if (j == 0)
        {
            Dvar = i;
            T = 'd';
        }

        else
        {

            if (d_t[j - 1] == d_s[i - 1])
            {
                Dvar = Avar;
                T = '-';
            }

            else
            {
                int l = d_s[i - 1] - 'a';
                int x = d_X[l * (n + 1) + j];
                Bvar = Dvar;

                if (x == 0)
                    Dvar = 1 + cudaMinimum(Avar, Bvar, i + j - 1, &T);
                else
                {
                    Cvar = d_D[(i - 1) * (n + 1) + x - 1];
                    Dvar = 1 + cudaMinimum(Avar, Bvar, Cvar + (j - 1 - x), &T);
                }
            }


        }

        //printf("i = %d, j = %d, Avar = %d, Bvar = %d, Cvar = %d, Dvar = %d \n",
        //    i, j, Avar, Bvar, Cvar, Dvar);

        d_D[i * (n + 1) + j] = Dvar;
        d_T[i * (n + 1) + j] = T;
    }

}

string CudaRetrieveTransformations(char* transformations, int m, int n)
{
    string path = "";
    int i = m, j = n;
    while (i > 0 && j > 0)
    {
        string k(1, transformations[i * (n + 1) + j]);
        path.append(k);
        switch (transformations[i * (n + 1) + j])
        {
        case 'd':
            i--;
            break;
        case 'i':
            j--;
            break;
        default:
            i--; j--;
            break;
        }
    }
    reverse(path.begin(), path.end());
    return path;
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

void printArray(char* array, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
            cout << array[i * n + j] << " ";
        cout << endl;
    }
    cout << endl << endl;
}

int CudaAlgorithm::LevenstheinDistance(string s, string t, string* transformPath)
{
    transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return tolower(c); });
    transform(t.begin(), t.end(), t.begin(), [](unsigned char c) { return tolower(c); });

    m = s.size(); n = t.size();

    cudaMalloc((void**)&d_s, (m + 1) * sizeof(char));
    cudaMalloc((void**)&d_t, (n + 1) * sizeof(char));
    cudaMalloc((void**)&d_X, q * (n + 1) * sizeof(int));
    cudaMalloc((void**)&d_D, (m + 1) * (n + 1) * sizeof(int));
    cudaMalloc((void**)&d_T, (m + 1) * (n + 1) * sizeof(char));

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

    ComputeD<<<1,n+1>>>(d_D, d_T, d_X, d_s, m, d_t, n, w);
    int* h_D = new int[(m + 1) * (n + 1) * sizeof(int)];
    cudaMemcpy(h_D, d_D, (m + 1) * (n + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    int dist = h_D[m * (n + 1) + n];
    printArray(h_D, m + 1, n + 1);

    char* h_T = new char[(m + 1) * (n + 1) * sizeof(char)];
    cudaMemcpy(h_T, d_T, (m + 1) * (n + 1) * sizeof(char), cudaMemcpyDeviceToHost);
    *transformPath = CudaRetrieveTransformations(h_T, m, n);
    printArray(h_T, m + 1, n + 1);

    cudaFree(d_T);
    cudaFree(d_D);
    cudaFree(d_X);
    cudaFree(d_t);
    cudaFree(d_s);
    delete[] h_s;
    delete[] h_t;
    delete[] h_T;
    delete[] h_D;

    return dist;
}

CudaAlgorithm::CudaAlgorithm()
{
    cudaMalloc((void**)&d_Q, (q + 1) * sizeof(char));
    cudaMemcpy(d_Q, h_Q, (q + 1) * sizeof(char), cudaMemcpyHostToDevice);
}

CudaAlgorithm::~CudaAlgorithm()
{
    cudaFree(d_Q);
}