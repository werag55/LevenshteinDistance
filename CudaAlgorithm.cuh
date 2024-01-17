#pragma once
#include <string>

using namespace std;

const int q = 26; // length of the alphabet
const int w = 32; // warps count

class CudaAlgorithm
{
	const char h_Q[q + 1] = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
							 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 
							 'w', 'x', 'y', 'z', '\0' };
	char* d_Q;

	char* d_s;
	int m;

	char* d_t;
	int n;

	int* d_X;
	int* d_D;
	char* d_T;
	
	int* d_nextColumn;
public:
	CudaAlgorithm();
	~CudaAlgorithm();
	int LevenstheinDistance(string s, string t, string* transformPath);
};