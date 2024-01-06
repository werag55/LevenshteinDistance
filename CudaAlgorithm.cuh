#pragma once
#include <string>

using namespace std;

const int q = 4; //27; // length of the alphabet + '\0'

class CudaAlgorithm
{
	/*const char h_Q[q] = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
					   'l', 'm', 'n', 'o', 'u', 'p', 'r', 's', 't', 'u', 'v', 
					   'w', 'x', 'y', 'z', '\0' };*/
	const char h_Q[q+1] = { 'A', 'C', 'G', 'T', '\0' };
	char* d_Q;

	char* d_s;
	int m;

	char* d_t;
	int n;

	int* d_X;
	int* d_D;
public:
	CudaAlgorithm();
	int LevenstheinDistance(string s, string t);
};