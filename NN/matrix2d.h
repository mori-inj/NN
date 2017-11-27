#ifndef __MATRIX2D__
#define __MATRIX2D__

#include <vector>
#include <assert.h>

#define LD long double

using namespace std;

template <typename T>
class Matrix2D
{
public:
	vector<vector<T> > data;
	int row, col;

	Matrix2D() {};
	Matrix2D(int row, int col);
	Matrix2D(vector<T> mat);
	Matrix2D(vector<vector<T> > mat);
	Matrix2D(const Matrix2D<T>& mat);

	void transpose(Matrix2D<T> mat);

	vector<T>& operator[] (const int idx) { return data[idx]; }
	const vector<T>& operator[] (const int idx) const { return data[idx]; }
};

#include "matrix2d.hpp"

#endif