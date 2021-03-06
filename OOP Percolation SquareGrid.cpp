#include <iostream> // Input/Output
#include <iomanip>  // for setw()
#include <cmath> // array calculations.
#include <ctime> // For random number generator time()
#include <vector> // I guess I'm using vectors insteaf of arrays.

using namespace std;

class ConnectionMatrix {

protected:
	int i, j, n;
	double num;
	int sqrtN;
	double edgeProbability;
	vector< vector<int>> graphEdges;

public:

	void setMatrixValues(int numVert, double edgeProbability, vector< vector<int>> _graphEdges);
};

void ConnectionMatrix::setMatrixValues(int numVert, double edgeProbability, vector< vector<int>> _graphEdges)
{
	i = 0;
	j = 0;
	n = numVert;
	sqrtN = (int)sqrt(n);
	graphEdges = _graphEdges;
	this->edgeProbability = edgeProbability;

	for (i; i < n; i++)
	{
		for (j = i; j < n; j++)// Starts from node i and then moves on from there so as to not double count.
		{
			if (j == i)
			{
				graphEdges[i][j] = -1;//No connect between the vertex it self.
				graphEdges[j][i] = -1;//No connect between the vertex it self.
			}

			num = rand() / (double)RAND_MAX;//This generates the random number.

			if ((i + 1) % sqrtN == 0 && (i != n - 1) && (j - i <= sqrtN) && (j - i != 0))// This handles the vertices that are the farthest horizontally.
			{//But skips the last one.
				if ((j >= i + 1) && (j <= i + sqrtN - 1))//The farthes node to the left in any row shoud have no connection except for maybe the one beloow it.
				{
					graphEdges[i][j] = -1;
					graphEdges[j][i] = -1;
				}

				else if (num < edgeProbability)//A connnection with a right most node of a row and the one below it.
				{
					graphEdges[i][j] = 1;
					graphEdges[j][i] = 1;
				}

				else//No connection with a right most node of a row and the one below it.
				{
					graphEdges[i][j] = 0;
					graphEdges[j][i] = 0;
				}
				continue;
			}

			if ((j - i <= sqrtN) && (j - i != 0))//Handles a node that is NOT the farthest to the right in any row.
			{
				if ((j >= i + 2) && (j - i <= sqrtN - 1))//A vertex distance 2 away or further horizontally, but not further than the square root of the number of verticies minus one. So really just before the one below it.
				{
					graphEdges[i][j] = -1;//There should not be a connection.
					graphEdges[j][i] = -1;//There should not be a connection.
				}

				else if (num < edgeProbability)// A connection with the node just to the right of it and/or also one just below it.
				{
					graphEdges[i][j] = 1;
					graphEdges[j][i] = 1;
				}

				else // No connection with the node just to the right of it and/or also one just below it.
				{
					graphEdges[i][j] = 0;
					graphEdges[j][i] = 0;
				}
			}

			else //An node below and to the right of the node just below it and the nodes below that row, there will be no connection.
			{
				graphEdges[i][j] = -1;
				graphEdges[j][i] = -1;
			}
		}
	}
}

class SquareMatrixVisual : public ConnectionMatrix {

public:

	void squareMatrixPrint();
};

void SquareMatrixVisual::squareMatrixPrint()
{
	i = n - 1;

	if (n <= 25) {//So the graph with the matrix with -1, 0, 1 doesn't over flow the screen.

		cout << "\n" << "Connections between Sites i(horizontal) vs. j(vertical):\n\n"; // Labeling this square matrix.

		for (i; i > -1; i--)
		{
			cout << "Site " << i;

			if (n < 10)
			{
				cout << ":";
			}

			else if ((n >= 10 && n <= 99) && i < 10)
			{
				cout << " :";
			}

			else
			{
				cout << ":";
			}

			for (int j = 0; j < n; j++)
			{
				if (graphEdges[i][j] == -1)
				{
					cout << "|(  )";
				}

				else
				{
					cout << "|( " << graphEdges[i][j] << ")";// This give the values of -1, no connectino at all; 0 no connection with probablity 1 - p
					// and finally 1, if there is a connection with probability p.
					if (graphEdges[i][j] == 1 || graphEdges[i][j] == 0)
					{
						//cout << " ";//This is just for proper spacing between the values.
					}
				}
			}

			cout << "|" << endl;//Move to the next line for the new node.
		}

		cout << setw(11); //Spacing to aline the node connections.
		for (int k = 0; k < n; k++)
		{
			if (k == n - 1)
			{
				cout << k;//Print until the last one.
				break;
			}

			cout << k << setw(5); //Just proper spacing.

			if (k == 9)
			{
				cout << setw(6);  // aligns a bit better
			}
		}
	}
	cout << endl;
}

class SquareLattice : public SquareMatrixVisual
{
public:
	
	void printSquareLattice();
};

void SquareLattice::printSquareLattice()
{
	cout << "\nVisual Representation of the square grid.\n\n";// space between the square matrix and the square grid.

	int k = 0;

	for (k; k < n; k++)
	{
		cout << "(" << k << ")"; //This should print out the node.

		int a = k + 1;

		if (a < n)
		{
			if (graphEdges[k][k + 1] == 1)//If the nodes next to each other horizontally are suppose to have an edge, then a proper length edge will be drawn.
			{
				if (k >= 100 && k <= 999)
					cout << "--";
				else if (k >= 10 && k <= 99)
					cout << "---";
				else
					cout << "----";
			}

			else//If the nodes next to each other horixontally are not suppose to have an edge, then proper spacing will be drawn.
			{
				if (k >= 100 && k <= 999)
					cout << "  ";
				else if (k >= 10 && k <= 99)
					cout << "   ";
				else
					cout << "    ";
			}
		}

		if ((k + 1) % sqrtN == 0 && k != n - 1)//Looking at the very last node on a "horizontal line" and not the very last node in the whole graph, we will put and edge where appropriate.
		{
			cout << endl;
			int m = 0;

			do {
				int l = sqrtN;// (int)sqrt(n);
				for (l; l > 0; l--)
				{
					if ((graphEdges[k - l + 1][k - l + 1 + sqrtN] == 1) && ((l - 1) % sqrtN != 0))//This is suppose to put a vertical edge between adjacent vertical vertices when appropriate.
					{
						cout << " |     ";
					}

					else if ((graphEdges[k - l + 1][k - l + 1 + sqrtN] == 1) && ((l - 1) % sqrtN == 0))//Puts a vertical edge between the right most vertex and the one below it when appropriate.
					{
						cout << " |" << endl;
						//break;
					}

					else if ((graphEdges[k - l + 1][k - l + 1 + sqrtN] == 0) && ((l - 1) % sqrtN == 0))
					{
						cout << endl;
					}
					
					else//Don't put a vertical edge between adjacent vertices when appropriate.
					{
						cout << "       ";
					}
				}

				m++;

			} while (m != 3);
		}
	}
	cout << "\n\n";//Just vertical spacing when program is done.
}

void askForInputs();

// Main function
int main() {

	askForInputs();

	return 0;
}

void askForInputs()
{
	int numberVert; //Number used for finite square lattice.
	int cNum; //Number taken in to be considered for the finite square lattice. Screen my overflow after cNum = 9
	int iNum; //Number of iterations with the same probability.
	double p; //Probability for the open edges. Note (1 - p) is the probability of the closed edges

	do
	{
		cout << endl;
		cout << "Please enter a number N such that the program will give an (N x N) size squre graph\n";
		cout << "and the number of iterations (i) with a space inbetween the two values\n";
		cout << "(positive whole numbers only):\n";
		cin >> cNum >> iNum;

		if (cin.fail())
		{
			cout << "\nError: Both have to be actual whole numbers.\n\n";
			cin.clear(); // Clears error.
			cin.ignore(1000, '\n'); // ignores 1000 space in the line and goes on to a new line...I think.
			continue;
		}

		if (cNum <= 0)
		{
			cout << "\nERROR: The number of nodes needs to be a positive integers.\n\n";
			continue;
		}

		if (iNum <= 0)
		{
			cout << "\nEROOR: The must be a positive number of iterations.\n\n";
			continue;
		}

		else {

			cout << "\nPlease give a number between 0 and 1 for the probability: ";
			cin >> p;

			while (p < 0 || p > 1) {
				cout << "This number is out of range. Enter a number between [0,1]: ";
				cin >> p;
			}

			numberVert = cNum * cNum;
		}

	} while (cNum <= 0 || iNum <= 0); // || cNum != 'Q' || cNum != 'q' || iNum != 'Q' || iNum != 'q');

	for (int i = 0; i < iNum; i++)
	{
		vector< vector<int>> tempVec(numberVert, vector<int>(numberVert));
		
		SquareLattice SMV;
		
		SMV.setMatrixValues(numberVert, p, tempVec);
		SMV.squareMatrixPrint();
		SMV.printSquareLattice();
	}
}