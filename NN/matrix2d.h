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

	vector<T>& operator[] (const int idx) { return data[idx]; }
	const vector<T>& operator[] (const int idx) const { return data[idx]; }
};

Matrix2D<LD> mat_mul(Matrix2D<LD> A, Matrix2D<LD> B)
{
	assert(A.col == B.row);
	int I = A.row;
	int J = B.row;
	int K = B.col;

	Matrix2D<LD> ret(I, K);
	for(int j=0; j<J; j++) {
		for(int i=0;i<I; i++) {
			ret[i][j] = 0;
		}

		for(int k=0; k<K; k++) {
			for(int i=0; i<I; i++) {
				ret[i][j] += A[i][k] * B[k][j];
			}
		}
	}

	return ret;
}

Matrix2D<LD> mul(Matrix2D<LD> A, Matrix2D<LD> B)
{
	assert(A.row == B.row);
	assert(A.col == 1);

	Matrix2D<LD> ret(B.row, B.col);
	for(int i=0; i<B.row; i++) {
		for(int j=0; j<B.col; j++) {
			ret[i][j] = A[i] * B[i][j];
		}
	}

	return ret;
}

#endif