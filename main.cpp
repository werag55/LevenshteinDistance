#include "SequentialAlgorithm.h"
#include "CudaAlgorithm.cuh"
#include <iostream>

int main(int argc, char* argv[])
{
	string s = "sitting", t = "kitten";


	SequentialAlgorithm seqAlg(2);
	int d = seqAlg.LevenstheinDistance(s, t);
	std::cout << d <<endl <<endl;

	CudaAlgorithm cudaAlg;
	int dc = cudaAlg.LevenstheinDistance(s, t);
	std::cout << dc;
}