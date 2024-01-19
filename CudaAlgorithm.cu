#include "CudaAlgorithm.cuh"
#include <iostream>
#include <algorithm>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <chrono>

using namespace std::chrono;

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)

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

//https://leimao.github.io/blog/Proper-CUDA-Error-Checking/
void checkLast(const char* const file, const int line)
{
    cudaError_t err{ cudaGetLastError() };
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << endl;
        cerr << cudaGetErrorString(err) << endl;
        exit(EXIT_FAILURE);
    }
}

/// <summary>
/// CUDA kernel for computing the X matrix for the Levenshtein distance algorithm.
/// </summary>
/// <param name="d_X">Device array for the X matrix from the algorithm.</param>
/// <param name="d_Q">Device array containing the accepted alphabet.</param>
/// <param name="q">Length of the alphabet.</param>
/// <param name="d_t">Device array for the target word.</param>
/// <param name="n">Length of the target word.</param>
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

/// <summary>
/// Calculates the minimum of three integers and assigns the corresponding transformation.
/// </summary>
/// <param name="a">First integer.</param>
/// <param name="b">Second integer.</param>
/// <param name="c">Third integer.</param>
/// <param name="t">Pointer to a character where the corresponding transformation will be stored.</param>
/// <returns>The minimum of the three integers.</returns>
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

/// <summary>
/// CUDA kernel for computing the D matrix and transformation path for the Levenshtein distance algorithm of a certain prefixes of 
/// input words.
/// </summary>
/// <param name="d_D">Device array for the distance matrix.</param>
/// <param name="d_T">Device array for the transformations matrix.</param>
/// <param name="d_X">Device array for the X matrix from the algorithm.</param>
/// <param name="d_s">Device array for the source word.</param>
/// <param name="m">Length of the source word.</param>
/// <param name="d_t">Device array for the target word.</param>
/// <param name="n">Length of the target word.</param>
/// <param name="w">Number of warps.</param>
/// <param name="d_nextColumn">Number of d_D / d_T column that the block should start computing with.</param>
__global__ void ComputeD(int* d_D, char* d_T, int* d_X,
    char* d_s, int m, char* d_t, int n, int w, int* d_nextColumn)
{

    int j = d_nextColumn[0] + threadIdx.x;

    // Shared memory allocation for d_t and a part (coresponding to the current block) of d_s
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

    // Initialize values for the first column
    Avar = 0;
    Dvar = j;
    d_D[j] = Dvar;
    d_T[j] = 'i';

    for (int i = 1; i <= m; i++)
    {

        __syncthreads();

        // Shuffle operation to get previous Dvar value
        int shflVal = __shfl_up(Dvar, 1);
        if (j % w == 0 && j != 0)
            Avar = d_D[(i - 1) * (n + 1) + j - 1];
        else
        {
            if (j != 0)
                Avar = shflVal;
        }
        // Calculate index for d_X matrix
        char c = shareds[i - 1]; //d_s[i - 1];
        int l = c - 'a';
        int x = d_X[l * (n + 1) + j];
        Cvar = d_D[(i - 1) * (n + 1) + x - 1];

        __syncthreads();

        if (j == 0) // // Update Dvar and T for the first column
        {
            Dvar = i;
            T = 'd';
        }

        else
        {

            if (/*d_t[j - 1]*/ sharedt[threadIdx.x] == c) // Matching characters, update Dvar and T accordingly
            {
                Dvar = Avar;
                T = '-';
            }


            else // Non-matching characters, update Dvar and T using cudaMinimum function
            {
                Bvar = Dvar;

                if (x == 0)
                    Dvar = 1 + cudaMinimum(Avar, Bvar, i + j - 1, &T);
                else
                    Dvar = 1 + cudaMinimum(Avar, Bvar, Cvar + (j - 1 - x), &T);
                
            }

        }

        // Update D and T matrices
        d_D[i * (n + 1) + j] = Dvar;
        d_T[i * (n + 1) + j] = T;
    }

    // Update d_nextColumn for the next runned block
    if (threadIdx.x == 0)
        d_nextColumn[0] += blockDim.x; 

    return;

}

// <summary>
/// Retrieves the transformation path from a 2D array of characters.
/// </summary>
/// <param name="transformations">2D array of characters representing the transformations.</param>
/// <param name="m">Size of the first dimension of the array.</param>
/// <param name="n">Size of the second dimension of the array.</param>
/// <returns>The transformation path as a string.</returns>
string CudaRetrieveTransformations(char* transformations, int m, int n)
{
    string path = "";
    int i = m, j = n;
    while (i != 0 || j != 0)
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

/// <summary>
/// Prints a 2D integer array with row and column labels for visualization, used while debugging to print distance array.
/// </summary>
/// <param name="array">Pointer to the 2D integer array.</param>
/// <param name="m">Number of rows in the array.</param>
/// <param name="n">Number of columns in the array.</param>
/// <param name="s">String representing row labels.</param>
/// <param name="t">String representing column labels.</param>
void PrintArray(int* array, int m, int n, string s, string t)
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

/// <summary>
/// Prints a 2D char array with row and column labels for visualization, used while debugging to print transformations array.
/// </summary>
/// <param name="array">Pointer to the 2D char array.</param>
/// <param name="m">Number of rows in the array.</param>
/// <param name="n">Number of columns in the array.</param>
/// <param name="s">String representing row labels.</param>
/// <param name="t">String representing column labels.</param>
void PrintArray(char* array, int m, int n, string s, string t)
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

/// <summary>
/// Calculates the Levenshtein distance between two words and determines the transformation path using a CUDA-based algorithm from
/// https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0186251
/// </summary>
/// <param name="s">The source word (containing only tle letters from Latin alphabet).</param>
/// <param name="t">The target word (containing only tle letters from Latin alphabet).</param>
/// /// <param name="transformPath">Pointer to a string where the transformation path will be stored.</param>
/// <returns>The Levenshtein distance between the source and target strings.</returns>
int CudaAlgorithm::LevenstheinDistance(string s, string t, string* transformPath)
{
    steady_clock::time_point start, stop;
    milliseconds duration;

    // Convert strings to lowercase
    transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return tolower(c); });
    transform(t.begin(), t.end(), t.begin(), [](unsigned char c) { return tolower(c); });

    // Set threads and blocks configuration
    m = (int)s.size(); n = (int)t.size();
    int threadsPerBlock = 512;
    if (threadsPerBlock > n + 1)
        threadsPerBlock = n + 1;
    int blocksPerGrid = ((n + 1) + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate device memory for arrays
    start = high_resolution_clock::now();
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_Q, (q + 1) * sizeof(char)));
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start);
    if (time)
        cout << "First CUDA function call duration: " << duration.count() << " ms" << endl << endl;
    start = high_resolution_clock::now();
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_s, (m + 1) * sizeof(char)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_t, (n + 1) * sizeof(char)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_X, q * (n + 1) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_D, (m + 1) * (n + 1) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_T, (m + 1) * (n + 1) * sizeof(char)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_nextColumn, sizeof(int)));
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start);
    if (time)
        cout << "Device memory allocation duration: " << duration.count() << " ms" << endl << endl;
    
    // Copy host data to device
    char* h_s = new char[m + 1];
    strcpy(h_s, s.c_str());
    char* h_t = new char[n + 1];
    strcpy(h_t, t.c_str());
    int h_nextColumn = 0;

    start = high_resolution_clock::now();
    CHECK_CUDA_ERROR(cudaMemcpy(d_Q, h_Q, (q + 1) * sizeof(char), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_s, h_s, (m + 1) * sizeof(char), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_t, h_t, (n + 1) * sizeof(char), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_nextColumn, &h_nextColumn, sizeof(int), cudaMemcpyHostToDevice));
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start);
    if (time)
        cout << "Data CPU-GPU copying duration: " << duration.count() << " ms" << endl << endl;

    // Compute X matrix on the device
    start = high_resolution_clock::now();
    ComputeX<<<1, q, (q + 1) * sizeof(char)>>>(d_X, d_Q, q, d_t, n);
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start);
    if (time)
        cout << "Computing X matrix duration: " << duration.count() << " ms" << endl << endl;

    // As the block responsible for calculating a given fragment of the d_D and d_T tablea uses the values calculated by the blocks 
    // responsible for earlier (in relation to the columns) fragments of the table, it is necessary to synchronize the calculations 
    // - therefore the blocks are run one by one in a loop.
    start = high_resolution_clock::now();
    for (int i = 0; i < blocksPerGrid; i++)
    {
        ComputeD <<<1, threadsPerBlock, (threadsPerBlock + m) * sizeof(char)>>>
            (d_D, d_T, d_X, d_s, m, d_t, n, w, d_nextColumn);
        cudaDeviceSynchronize();
        CHECK_LAST_CUDA_ERROR();
    }
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start);
    if (time)
        cout << "Computing distance and transformations arrays duration: " << duration.count() << " ms" << endl << endl;

    // Copy results back to host
    start = high_resolution_clock::now();
    int* h_D = new int[(m + 1) * (n + 1) * sizeof(int)];
    char* h_T = new char[(m + 1) * (n + 1) * sizeof(char)];
    CHECK_CUDA_ERROR(cudaMemcpy(h_D, d_D, (m + 1) * (n + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_T, d_T, (m + 1) * (n + 1) * sizeof(char), cudaMemcpyDeviceToHost));
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start);
    if (time)
        cout << "Data GPU-CPU copying duration: " << duration.count() << " ms" << endl << endl;

    // Retrieve Levenshtein distance and transformation path
    int dist = h_D[m * (n + 1) + n];
    *transformPath = CudaRetrieveTransformations(h_T, m, n);

    // Print distance and transformations arrays
    //PrintArray(h_D, m + 1, n + 1, s, t);
    //PrintArray(h_T, m + 1, n + 1, s, t);

    // Clean up memory
    CHECK_CUDA_ERROR(cudaFree(d_Q));
    CHECK_CUDA_ERROR(cudaFree(d_T));
    CHECK_CUDA_ERROR(cudaFree(d_D));
    CHECK_CUDA_ERROR(cudaFree(d_X));
    CHECK_CUDA_ERROR(cudaFree(d_t));
    CHECK_CUDA_ERROR(cudaFree(d_s));
    CHECK_CUDA_ERROR(cudaFree(d_nextColumn));
    delete[] h_s;
    delete[] h_t;
    delete[] h_T;
    delete[] h_D;

    return dist;
}

CudaAlgorithm::CudaAlgorithm(bool time)
{
    this->time = time;
}

