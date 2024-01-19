#pragma once
#include <string>

using namespace std;

const int q = 26; // Length of the alphabet
const int w = 32; // Warps count

class CudaAlgorithm
{
	bool time; // Boolean indicating whether the time option is selected

	const char h_Q[q + 1] = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', // Array containing the accepted alphabet
							 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 
							 'w', 'x', 'y', 'z', '\0' };
	char* d_Q; // Device array containing the accepted alphabet

	char* d_s; // Device array for the source word
	int m; // Length of the source word

	char* d_t; // Device array for the target word
	int n; // Length of the target word

	int* d_X; // Device array containing the X matrix from the algorithm 
	int* d_D; // Device array containing distances beetwen all prefixes of the input words
	char* d_T; // Device array containig trasnfromation path beetwen all prefixes of the input words
	
	int* d_nextColumn; // Number of d_D / d_T column that the next runned block should start computing with
public:
	CudaAlgorithm(bool time);
	int LevenstheinDistance(string s, string t, string* transformPath);
};