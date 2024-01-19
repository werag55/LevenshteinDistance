#pragma once
#include <string>

using namespace std;

class SequentialAlgorithm
{
	bool time; // Boolean indicating whether the time option is selected
public:
	SequentialAlgorithm(bool time);
	int LevenstheinDistance(string s, string t, string* transformPath);
};