#include "matrix2d.h"

template <typename T>
Matrix2D<T>::Matrix2D(int row, int col)
{
	for(int i=0; i<row; i++) {
		data.push_back(vector<T>());
		for(int j=0; j<col; j++) {
			data[i].push_back(T());
		}
	}
}

template <typename T>
Matrix2D<T>::Matrix2D(vector<T> mat)
{
	int row = (int)mat.size();
	int col = 1;

	for(int i=0; i<row; i++) {
		data.push_back(vector<T>());
		data[i].push_back(mat[i]);
	}
}

template <typename T>
Matrix2D<T>::Matrix2D(vector<vector<T> > mat)
{
	int row = (int)mat.size();
	int col = (int)mat[0].size();
	for(int i=0; i<row; i++) {
		assert(col == (int)mat[i].size());
	}

	for(int i=0; i<row; i++) {
		data.push_back(vector<T>());
		for(int j=0; j<col; j++) {
			data[i].push_back(mat[i][j]);
		}
	}
}

template <typename T>
Matrix2D<T>::Matrix2D(const Matrix2D<T>& mat)
{
	int row = (int)mat.row;
	int col = (int)mat.col;

	for(int i=0; i<row; i++) {
		data.push_back(vector<T>());
		for(int j=0; j<col; j++) {
			data[i].push_back(mat[i][j]);
		}
	}
}
