#include "SequentialAlgorithm.h"
#include "CudaAlgorithm.cuh"
#include <iostream>
#include <vector>
#include <chrono>

using namespace std;
using namespace std::chrono;

void ManageOptions(int argc, char* argv[], bool* cuda, bool* seq, bool* time, string* s, string* t);

string GetOption(const vector<string>& args, const string& option_name);

bool HasOption(const vector<string>& args, const string& option_name);

void PrintUsage(string fullPath);

int main(int argc, char* argv[])
{
	steady_clock::time_point start, stop;
	milliseconds duration;

    bool cuda = true, seq = true, time = false;
	string s = "sitting", t = "kitten", transformations = "";

	ManageOptions(argc, argv, &cuda, &seq, &time, &s, &t);

	if (seq)
	{
		cout << "-------------------SequentialAlgorithm-------------------" << endl;
		start = high_resolution_clock::now();
		SequentialAlgorithm seqAlg;
		int d = seqAlg.LevenstheinDistance(s, t, &transformations);
		cout << "Distance between \"" << s << "\" and \"" << t << "\": " << d << endl;
		cout << "Transformations : " << transformations << endl;		
		stop = high_resolution_clock::now();
		duration = duration_cast<milliseconds>(stop - start);
		cout << "SequentialAlgorithm duration " << duration.count() << " ms" << endl <<endl;

	}

	if (cuda)
	{
		cout << "----------------------CudaAlgorithm----------------------" << endl;
		start = high_resolution_clock::now();
		CudaAlgorithm cudaAlg;
		int dc = cudaAlg.LevenstheinDistance(s, t, &transformations);
		cout << "Distance between \"" << s << "\" and \"" << t << "\": " << dc << endl;
		cout << "Transformations : " << transformations << endl;
		stop = high_resolution_clock::now();
		duration = duration_cast<milliseconds>(stop - start);
		cout << "CudaAlgorithm duration " << duration.count() << " ms" << endl << endl;
	}
}

string GetOption(const vector<string>& args, const string& option_name)
{
    for (auto it = args.begin(), end = args.end(); it != end; ++it)
    {
        if (*it == option_name)
            if (it + 1 != end)
                return *(it + 1);
    }

    return "";
}

bool HasOption(const vector<string>& args, const string& option_name)
{
    for (auto it = args.begin(), end = args.end(); it != end; ++it)
    {
        if (*it == option_name)
            return true;
    }

    return false;
}

void PrintUsage(string fullPath)
{
	size_t lastSlash = fullPath.find_last_of("/\\");
	string programName = (lastSlash != std::string::npos) ? fullPath.substr(lastSlash + 1) : fullPath;
	programName = programName.c_str();
	cout << "usage";
}

void ManageOptions(int argc, char* argv[], bool* cuda, bool* seq, bool* time, string* s, string* t)
{
	const vector<string> args(argv, argv + argc);
	if (HasOption(args, "-h") || HasOption(args, "--help"))
	{
		PrintUsage(argv[0]);
		return;
	}

	if ((HasOption(args, "-c") || HasOption(args, "--cuda"))
		&& (HasOption(args, "-sq") || HasOption(args, "--sequential")))
	{
		cout << "You cannot use the --cuda (-c) and --sequential (-sq) options simultaneously. \n"
			<< "If you want to run both algorithms, simply drop both of these options, as \n"
			<< "this is the default configuration. \n";
		PrintUsage(argv[0]);
		return;
	}

	if (HasOption(args, "-c") || HasOption(args, "--cuda"))
	{
		*cuda = true;
		*seq = false;
	}

	if (HasOption(args, "-sq") || HasOption(args, "--sequential"))
	{
		*cuda = false;
		*seq = true;
	}

	if (HasOption(args, "-tm") || HasOption(args, "--time"))
		*time = true;

	if (HasOption(args, "-s"))
		*s = GetOption(args, "-s");

	if (HasOption(args, "-t"))
		*t = GetOption(args, "-t");
}