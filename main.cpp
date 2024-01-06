#include "SequentialAlgorithm.h"
#include "CudaAlgorithm.cuh"
#include <iostream>

int main()
{
	//SequentialAlgorithm seqAlg(2);
	//int d = seqAlg.LevenstheinDistance("kitten", "sitting");
	//std::cout << d;

	CudaAlgorithm cudaAlg;
	int d = cudaAlg.LevenstheinDistance("sitting", "kitten");
	std::cout << d;
}