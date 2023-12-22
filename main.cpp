#include "SequentialAlgorithm.h"
#include <iostream>

int main()
{
	SequentialAlgorithm seqAlg(2);
	int d = seqAlg.LevenstheinDistance("kitten", "sitting");
	std::cout << d;
}