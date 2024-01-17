#include "SequentialAlgorithm.h"
#include <iostream>

SequentialAlgorithm::SequentialAlgorithm()
{}

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

string RetrieveTransformations(char** transformations, int m, int n)
{
	string path = "";
	int i = m, j = n;
	while (i > 0 && j > 0)
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

int SequentialAlgorithm::LevenstheinDistance(string s, string t, string* transformPath)
{
	char transform;
	int d;
	int m = s.size(), n = t.size();

	int** distance = new int* [m + 1];
	char** transformations = new char* [m + 1];
	for (int i = 0; i <= m; i++)
	{
		distance[i] = new int[n + 1];
		transformations[i] = new char[n + 1];
	}		

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

	/*cout << "    ";
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
			cout << distance[i][j] << " ";
		cout << endl;
	}
	cout << endl << endl;

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
			cout << transformations[i][j] << " ";
		cout << endl;
	}
	cout << endl << endl;*/

	*transformPath = RetrieveTransformations(transformations, m, n);
	d = distance[m][n];

	for (int i = 0; i < m; i++)
	{
		delete[] distance[i];
		delete[] transformations[i];
	}
	delete[] distance;
	delete[] transformations;

	return d;
}
