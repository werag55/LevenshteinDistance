#include "CudaAlgorithm.cuh"
#include <iostream>
#include <algorithm>
#include <cuda.h>
//#include <cuda/barrier>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define __CUDACC__
#include <cooperative_groups.h>
#include <cuda_runtime_api.h> 

using namespace cooperative_groups;

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

__device__ void clock_block(int* d_o, clock_t clock_count)
{
    clock_t start_clock = clock();
    clock_t clock_offset = 0;
    while (clock_offset < clock_count)
    {
        clock_offset = clock() - start_clock;
    }
    *d_o = (int)clock_offset;
}

__global__ void ComputeD(int* d_D, char* d_T, int* d_X,
    char* d_s, int m, char* d_t, int n, int w, int* d_nextColumn)
{

    //int j = blockIdx.x * blockDim.x + threadIdx.x
    //int nextColumn = atomic_add(d_nextColumn[0], );
    int j = d_nextColumn[0] + threadIdx.x;
    //printf(" d_nextColumn[0] = %d, threadIdx.x = % d\n", d_nextColumn[0], threadIdx.x);

    extern __shared__ char sharedt[];
    char* shareds = sharedt + blockDim.x * sizeof(char);
    if (j != 0 && j <= n)
        sharedt[threadIdx.x] = d_t[j - 1];
    int fragmentSize = (m + blockDim.x) / blockDim.x;
    int startIndex = threadIdx.x * fragmentSize;
    for (int i = 0; i < fragmentSize && startIndex + i < m; i++)
        shareds[startIndex + i] = d_s[startIndex + i];
    __syncthreads();

    if (j > n)
        return;

    int Dvar, Avar, Bvar, Cvar;
    char T;

    Avar = 0;
    Dvar = j;
    d_D[j] = Dvar;
    d_T[j] = 'i';

    for (int i = 1; i <= m; i++)
    {

        __syncthreads();

        int shflVal = __shfl_up(Dvar, 1);
        if (j % w == 0 && j != 0)
            Avar = d_D[(i - 1) * (n + 1) + j - 1];
        else
        {
            if (j != 0)
                Avar = shflVal;
        }
        char c = shareds[i - 1]; //d_s[i - 1];
        int l = c - 'a';
        int x = d_X[l * (n + 1) + j];
        Cvar = d_D[(i - 1) * (n + 1) + x - 1];


        __syncthreads();

        if (j == 0)
        {
            Dvar = i;
            T = 'd';
        }

        else
        {

            if (/*d_t[j - 1]*/ sharedt[threadIdx.x] == c)
            {
                Dvar = Avar;
                T = '-';
            }


            else
            {
                Bvar = Dvar;

                if (x == 0)
                    Dvar = 1 + cudaMinimum(Avar, Bvar, i + j - 1, &T);
                else
                    Dvar = 1 + cudaMinimum(Avar, Bvar, Cvar + (j - 1 - x), &T);
                
            }

        }

        d_D[i * (n + 1) + j] = Dvar;
        d_T[i * (n + 1) + j] = T;
    }

    if (threadIdx.x == 0)
    {
        //printf("Adding %d to %d\n", blockDim.x, d_nextColumn[0]);
        d_nextColumn[0] += blockDim.x;
        //printf("Result %d\n", d_nextColumn[0]);
    }
    //printf("Theras %d succes\n", j);
    return;

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

void printArray(int* array, int m, int n, string s, string t)
{
    cout << "    ";
    for (int j = 0; j < n - 1; j++)
        cout << t[j] << " ";
    cout << endl;
    for (int i = 0; i < m; i++)
    {
        if (i == 0)
            cout << "  ";
        else
            cout << s[i - 1] << " ";

        for (int j = 0; j < n; j++)
            cout << array[i * n + j] << " ";
        cout << endl;
    }
    cout << endl << endl;
}

void printArray(char* array, int m, int n, string s, string t)
{
    cout << "    ";
    for (int j = 0; j < n - 1; j++)
        cout << t[j] << " ";
    cout << endl;
    for (int i = 0; i < m; i++)
    {
        if (i == 0)
            cout << "  ";
        else
            cout << s[i - 1] << " ";

        for (int j = 0; j < n; j++)
            cout << array[i * n + j] << " ";
        cout << endl;
    }
    cout << endl << endl;
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

    m = (int)s.size(); n = (int)t.size();
    int threadsPerBlock = 512;
    if (threadsPerBlock > n + 1)
        threadsPerBlock = n + 1;
    int blocksPerGrid = ((n + 1) + threadsPerBlock - 1) / threadsPerBlock;

    cudaMalloc((void**)&d_s, (m + 1) * sizeof(char));
    cudaMalloc((void**)&d_t, (n + 1) * sizeof(char));
    cudaMalloc((void**)&d_X, q * (n + 1) * sizeof(int));
    cudaMalloc((void**)&d_D, (m + 1) * (n + 1) * sizeof(int));
    cudaMalloc((void**)&d_T, (m + 1) * (n + 1) * sizeof(char));

    cudaMalloc((void**)&d_nextColumn, sizeof(int));
    /*int* h_nextColumn = new int[1];
    h_nextColumn[0] = 0;*/
    int h_nextColumn = 0;
    cudaMemcpy(d_nextColumn, &h_nextColumn, sizeof(int), cudaMemcpyHostToDevice);

    char* h_s = new char[m + 1];
    strcpy(h_s, s.c_str());
    char* h_t = new char[n + 1];
    strcpy(h_t, t.c_str());
    cudaMemcpy(d_s, h_s, (m + 1) * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_t, h_t, (n + 1) * sizeof(char), cudaMemcpyHostToDevice);

    ComputeX<<<1, q, (q + 1) * sizeof(char)>>>(d_X, d_Q, q, d_t, n);
    int* h_X = new int[q * (n + 1)];
    cudaMemcpy(h_X, d_X, q * (n + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    //printArray(h_X, q, n + 1);

   /* ComputeD<<<blocksPerGrid, threadsPerBlock, (threadsPerBlock + m) * sizeof(char) >>>
        (d_D, d_T, d_X, d_s, m, d_t, n, w);
    cudaDeviceSynchronize();*/

    for (int i = 0; i < blocksPerGrid; i++)
    {
        ComputeD <<<1, threadsPerBlock, (threadsPerBlock + m) * sizeof(char)>>>
            (d_D, d_T, d_X, d_s, m, d_t, n, w, d_nextColumn);
        cudaDeviceSynchronize();

        //int* h_D = new int[(m + 1) * (n + 1) * sizeof(int)];
        //cudaMemcpy(h_D, d_D, (m + 1) * (n + 1) * sizeof(int), cudaMemcpyDeviceToHost);
        //printArray(h_D, m + 1, n + 1, s, t);

        //char* h_T = new char[(m + 1) * (n + 1) * sizeof(char)];
        //cudaMemcpy(h_T, d_T, (m + 1) * (n + 1) * sizeof(char), cudaMemcpyDeviceToHost);
        ////*transformPath = CudaRetrieveTransformations(h_T, m, n);
        //printArray(h_T, m + 1, n + 1, s, t);

        cudaError_t err = cudaGetLastError();

        if (err != cudaSuccess)
        {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
            //return -1;
            // Possibly: exit(-1) if program cannot continue....
        }
        //cudaMemcpy(h_nextColumn, &d_nextColumn, sizeof(int), cudaMemcpyDeviceToHost);
        //cout << h_nextColumn[0];
    }
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        //return -1;
        // Possibly: exit(-1) if program cannot continue....
    }


    int* h_D = new int[(m + 1) * (n + 1) * sizeof(int)];
    cudaMemcpy(h_D, d_D, (m + 1) * (n + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    int dist = h_D[m * (n + 1) + n];
    //printArray(h_D, m + 1, n + 1, s, t);

    char* h_T = new char[(m + 1) * (n + 1) * sizeof(char)];
    cudaMemcpy(h_T, d_T, (m + 1) * (n + 1) * sizeof(char), cudaMemcpyDeviceToHost);
    *transformPath = CudaRetrieveTransformations(h_T, m, n);
    //printArray(h_T, m + 1, n + 1, s, t);

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

