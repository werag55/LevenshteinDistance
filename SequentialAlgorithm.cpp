#include "SequentialAlgorithm.h"
#include <iostream>
#include <chrono>

using namespace std::chrono;

SequentialAlgorithm::SequentialAlgorithm(bool time)
{
	this->time = time;
}

/// <summary>
/// Calculates the minimum of three integers and assigns the corresponding transformation.
/// </summary>
/// <param name="a">First integer.</param>
/// <param name="b">Second integer.</param>
/// <param name="c">Third integer.</param>
/// <param name="t">Pointer to a character where the corresponding transformation will be stored.</param>
/// <returns>The minimum of the three integers.</returns>
int minimum(int a, int b, int c, char* t)
{
	int min = a;
	*t = 's';

	if (b < min)
	{
		min = b;
		*t = 'i';
	}

	if (c < min)
	{
		min = c;
		*t = 'd';
	}

	return min;
}

// <summary>
/// Retrieves the transformation path from a 2D array of characters.
/// </summary>
/// <param name="transformations">2D array of characters representing the transformations.</param>
/// <param name="m">Size of the first dimension of the array.</param>
/// <param name="n">Size of the second dimension of the array.</param>
/// <returns>The transformation path as a string.</returns>
string RetrieveTransformations(char** transformations, int m, int n)
{
	string path = "";
	int i = m, j = n;
	while (i != 0 || j != 0)
	{
		string k(1, transformations[i][j]);
		path.append(k);
		switch (transformations[i][j])
		{
		case 'd':
			i--;
			break;
		case 'i':
			j--;
			break;
		default:
			i--; j--;
			break;
		}
	}
	reverse(path.begin(), path.end());
	return path;
}

/// <summary>
/// Prints a 2D integer array with row and column labels for visualization, used while debugging to print distance array.
/// </summary>
/// <param name="array">Pointer to the 2D integer array.</param>
/// <param name="m">Number of rows in the array.</param>
/// <param name="n">Number of columns in the array.</param>
/// <param name="s">String representing row labels.</param>
/// <param name="t">String representing column labels.</param>
void SeqPrintArray(int** array, int m, int n, string s, string t)
{
	cout << "    ";
	for (int j = 0; j < n; j++)
		cout << t[j] << " ";
	cout << endl;
	for (int i = 0; i <= m; i++)
	{
		if (i == 0)
			cout << "  ";
		else
			cout << s[i - 1] << " ";

		for (int j = 0; j <= n; j++)
			cout << array[i][j] << " ";
		cout << endl;
	}
	cout << endl << endl;
}

/// <summary>
/// Prints a 2D char array with row and column labels for visualization, used while debugging to print transformations array.
/// </summary>
/// <param name="array">Pointer to the 2D char array.</param>
/// <param name="m">Number of rows in the array.</param>
/// <param name="n">Number of columns in the array.</param>
/// <param name="s">String representing row labels.</param>
/// <param name="t">String representing column labels.</param>
void SeqPrintArray(char** array, int m, int n, string s, string t)
{
	cout << "    ";
	for (int j = 0; j < n; j++)
		cout << t[j] << " ";
	cout << endl;
	for (int i = 0; i <= m; i++)
	{
		if (i == 0)
			cout << "  ";
		else
			cout << s[i - 1] << " ";

		for (int j = 0; j <= n; j++)
			cout << array[i][j] << " ";
		cout << endl;
	}
	cout << endl << endl;
}

/// <summary>
/// Calculates the Levenshtein distance between two words and determines the transformation path.
/// </summary>
/// <param name="s">The source word.</param>
/// <param name="t">The target word.</param>
/// <param name="transformPath">Pointer to a string where the transformation path will be stored.</param>
int SequentialAlgorithm::LevenstheinDistance(string s, string t, string* transformPath)
{
	char transform;
	int d;
	int m = s.size(), n = t.size();

	// Create 2D arrays for distance and transformations
	int** distance = new int* [m + 1];
	char** transformations = new char* [m + 1];
	for (int i = 0; i <= m; i++)
	{
		distance[i] = new int[n + 1];
		transformations[i] = new char[n + 1];
	}

	steady_clock::time_point start, stop;
	milliseconds duration;
	start = high_resolution_clock::now();

	// Initialize base cases for the dynamic programming approach		
	for (int i = 0; i <= m; i++)
	{
		distance[i][0] = i;
		transformations[i][0] = 'd';
	}
	for (int j = 1; j <= n; j++)
	{
		distance[0][j] = j;
		transformations[0][j] = 'i';
	}

	// Populate the distance and transformations matrices
	for (int j = 1; j <= n; j++)
	{
		for (int i = 1; i <= m; i++)
		{
			int notMatching = s[i - 1] == t[j - 1] ? 0 : 1;

			distance[i][j] = minimum(distance[i - 1][j - 1] + notMatching, // substitution
									 distance[i][j - 1] + 1,			   // insertion
									 distance[i - 1][j] + 1,               // deletion
									 &transform);

			if (transform == 's' && notMatching == 0)
				transform = '-';

			transformations[i][j] = transform;
		}
	}

	stop = high_resolution_clock::now();
	duration = duration_cast<milliseconds>(stop - start);
	if (time)
		cout << "Computing distance and transformations arrays duration: " << duration.count() << " ms" << endl << endl;

	// Print distance and transformations arrays
	//SeqPrintArray(distance, m, n, s, t);
	//SeqPrintArray(transformations, m, n, s, t);

	// Retrieve the transformation path and Levenshtein distance
	*transformPath = RetrieveTransformations(transformations, m, n);
	d = distance[m][n];

	// Clean up memory
	for (int i = 0; i < m; i++)
	{
		delete[] distance[i];
		delete[] transformations[i];
	}
	delete[] distance;
	delete[] transformations;

	return d;
}
