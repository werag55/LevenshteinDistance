#include "SequentialAlgorithm.h"
#include "CudaAlgorithm.cuh"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>

using namespace std;
using namespace std::chrono;

int ManageOptions(int argc, char* argv[], bool* cuda, bool* seq, bool* time, bool* hide, string* s, string* t);

string GetOption(const vector<string>& args, const string& option_name);

bool HasOption(const vector<string>& args, const string& option_name);

void PrintUsage(string fullPath);

int main(int argc, char* argv[])
{
	steady_clock::time_point start, stop;
	milliseconds duration;

    bool cuda = true, seq = true, time = false, hide = false;
	string s = "karnisz", t = "dokarmia", transformations = "";

	if (ManageOptions(argc, argv, &cuda, &seq, &time, &hide, &s, &t) == -1)
	{
		PrintUsage(argv[0]);
		return 0;
	}

	if (!hide)
		cout << "Source word: " << s << endl << "Target word: " << t << endl << endl;

	if (seq)
	{
		cout << "-------------------SequentialAlgorithm-------------------" << endl;
		SequentialAlgorithm seqAlg(time);
		start = high_resolution_clock::now();
		int d = seqAlg.LevenstheinDistance(s, t, &transformations);
		cout << "Distance: " << d << endl;
		if (!hide)
			cout << "Transformations : " << transformations << endl;		
		stop = high_resolution_clock::now();
		duration = duration_cast<milliseconds>(stop - start);
		cout << "SequentialAlgorithm duration: " << duration.count() << " ms" << endl <<endl;

	}

	if (cuda)
	{
		cout << "----------------------CudaAlgorithm----------------------" << endl;
		CudaAlgorithm cudaAlg(time);
		start = high_resolution_clock::now();
		int dc = cudaAlg.LevenstheinDistance(s, t, &transformations);
		cout << "Distance: " << dc << endl;
		if (!hide)
			cout << "Transformations : " << transformations << endl;
		stop = high_resolution_clock::now();
		duration = duration_cast<milliseconds>(stop - start);
		cout << "CudaAlgorithm duration: " << duration.count() << " ms" << endl << endl;
	}
}

/// <summary>
/// Gets the value of a specified option from the command line arguments.
/// </summary>
/// <param name="args">Vector containing the command line arguments.</param>
/// <param name="option_name">Name of the option to retrieve.</param>
/// <returns>The value of the specified option or an empty string if the option is not found.</returns>
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

/// <summary>
/// Checks if a specified option is present in the command line arguments.
/// </summary>
/// <param name="args">Vector containing the command line arguments.</param>
/// <param name="option_name">Name of the option to check.</param>
/// <returns>True if the option is present, false otherwise.</returns>
bool HasOption(const vector<string>& args, const string& option_name)
{
    for (auto it = args.begin(), end = args.end(); it != end; ++it)
    {
        if (*it == option_name)
            return true;
    }

    return false;
}

/// <summary>
/// Prints the usage information for the program.
/// </summary>
/// <param name="fullPath">Full path of the program executable.</param>
void PrintUsage(string fullPath)
{
	size_t lastSlash = fullPath.find_last_of("/\\");
	string programName = (lastSlash != std::string::npos) ? fullPath.substr(lastSlash + 1) : fullPath;
	programName = programName.c_str();

	cout << "Usage: " << programName << " [OPTIONS]\n\n"
		<< "Options:\n"
		<< "  -h, --help              Display this usage information.\n"
		<< "  -c, --cuda              Run only the algorithm using CUDA.\n"
		<< "  -sq, --sequential       Run only the sequential algorithm.\n"
		<< "  -tm, --time              Display detailed time measurements.\n"
		<< "  -s WORD, --source WORD  Specify the input word for the Levenshtein algorithm.\n"
		<< "  -t WORD, --target WORD  Specify the target word for the Levenshtein algorithm.\n"
		<< "  -f FILE, --file FILE    Specify the filename of the file containing words for the algorith\n" 
		<< "						  (source word in the first line, target word in the second line). \n"
		<< "						  The file should be place in the same directory as the .exe file. \n"
		<< "                          (If used simultaneously with -s or -t, only -f will be considered.)\n"
		<< "  -hd, --hide             Specify if you want to hide the resulting path between words, i.e., not to be displayed in the console.\n";
	
}

/// <summary>
/// Manages command line options and updates corresponding variables accordingly.
/// </summary>
/// <param name="argc">Number of command line arguments.</param>
/// <param name="argv">Array of command line arguments.</param>
/// <param name="cuda">Pointer to a boolean indicating whether the CUDA option is selected.</param>
/// <param name="seq">Pointer to a boolean indicating whether the sequential option is selected.</param>
/// <param name="time">Pointer to a boolean indicating whether the time option is selected.</param>
/// <param name="hide">Pointer to a boolean indicating whether the hide option is selected.</param>
/// <param name="s">Pointer to a string containing the source word for the Levenshtein algorithm.</param>
/// <param name="t">Pointer to a string containing the target word for the Levenshtein algorithm.</param>
/// <returns>0 on success, -1 on error.</returns>
int ManageOptions(int argc, char* argv[], bool* cuda, bool* seq, bool* time, bool* hide, string* s, string* t)
{
	const vector<string> args(argv, argv + argc);

	for (const string& arg : args) {
		if (arg.size() > 0 && arg[0] == '-' && arg != "-c" && arg != "--cuda" && arg != "-sq" && arg != "--sequential" &&
			arg != "-tm" && arg != "--time" && arg != "-s" && arg != "--source" && arg != "-t" && arg != "--target" &&
			arg != "-f" && arg != "--file" && arg != "-hd" && arg != "--hide" && arg != "-h" && arg != "--help") {
			cout << "ERROR: Unrecognized option: " << arg << "\n";
			return -1;
		}
	}

	if (HasOption(args, "-h") || HasOption(args, "--help"))
		return -1;

	if ((HasOption(args, "-c") || HasOption(args, "--cuda"))
		&& (HasOption(args, "-sq") || HasOption(args, "--sequential")))
	{
		cout << "ERROR: You cannot use the --cuda (-c) and --sequential (-sq) options simultaneously. \n"
			 << "If you want to run both algorithms, simply drop both of these options, as \n"
			 << "this is the default configuration. \n";
		return -1;
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
	else if (HasOption(args, "--source"))
		*s = GetOption(args, "--source");

	if (HasOption(args, "-t"))
		*t = GetOption(args, "-t");
	else if (HasOption(args, "--target"))
		*t = GetOption(args, "--target");

	if (HasOption(args, "-f") || HasOption(args, "--file"))
	{
		if (HasOption(args, "-s") || HasOption(args, "--source") 
			|| HasOption(args, "-t") || HasOption(args, "--target"))
			cout << "WARNING: Since you used the --file (-f) option, the arguments given for \n"
				 << "the - s and -t options will be ignored. \n";
		
		string fileName = "";
		if (HasOption(args, "-f"))
			fileName = GetOption(args, "-f");
		else 
			fileName = GetOption(args, "--file");

		if (fileName.empty() || fileName == "")
		{
			cout << "ERROR: You need to provide file name for the --file (-f) option. \n";
			return -1;
		}

		try
		{
			steady_clock::time_point start, stop;
			milliseconds duration;
			start = high_resolution_clock::now();

			ifstream file(".\\" + fileName);
			if (!file.is_open())
				throw runtime_error("File not found\n");

			string line;
			getline(file, line);
			istringstream iss(line);
			iss >> *s;
			getline(file, line);
			iss = istringstream(line);
			iss >> *t;

			stop = high_resolution_clock::now();
			duration = duration_cast<milliseconds>(stop - start);
			if(*time)
				cout << "Loading data from file duration " << duration.count() << " ms" << endl << endl;

		}
		catch (...)
		{
			cout << "Could not read the data\n";
			return -1;
		}

	}

	if (HasOption(args, "-hd") || HasOption(args, "--hide"))
		*hide = true;

	return 0;
}