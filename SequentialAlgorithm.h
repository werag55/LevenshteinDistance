#pragma once
#include <string>

using namespace std;

class SequentialAlgorithm
{
public:
	SequentialAlgorithm();
	int LevenstheinDistance(string s, string t, string* transformPath);
};