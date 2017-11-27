template <typename T>
Matrix2D<T>::Matrix2D(int row, int col)
{
	this->row = row;
	this->col = col;
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
	row = (int)mat.size();
	col = 1;

	for(int i=0; i<row; i++) {
		data.push_back(vector<T>());
		data[i].push_back(mat[i]);
	}
}

template <typename T>
Matrix2D<T>::Matrix2D(vector<vector<T> > mat)
{
	row = (int)mat.size();
	col = (int)mat[0].size();
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
	row = (int)mat.row;
	col = (int)mat.col;

	for(int i=0; i<row; i++) {
		data.push_back(vector<T>());
		for(int j=0; j<col; j++) {
			data[i].push_back(mat[i][j]);
		}
	}
}

template <typename T>
void Matrix2D<T>::transpose(Matrix2D<T> mat)
{
	row = mat.col;
	col = mat.row;

	data = vector<vector<T> >(row);

	for(int i=0; i<row; i++) {
		for(int j=0; j<col; j++) {
			data[i].push_back(mat[j][i]);
		}
	}
}

Matrix2D<LD> mat_mul(Matrix2D<LD> A, Matrix2D<LD> B)
{
	assert(A.col == B.row);
	int I = A.row;
	int J = B.row;
	int K = B.col;

	Matrix2D<LD> ret(I, K);
	for(int i=0; i<I; i++) {
		for(int k=0; k<K; k++) {
			ret[i][k] = 0;
			for(int j=0; j<J; j++) {
				ret[i][k] += A[i][j] * B[j][k];
			}
		}
	}

	return ret;
}

Matrix2D<LD> mul(Matrix2D<LD> A, Matrix2D<LD> B)
{
	assert(A.row == B.col);
	assert(A.col == 1);

	Matrix2D<LD> ret(B.row, B.col);
	for(int i=0; i<B.row; i++) {
		for(int j=0; j<B.col; j++) {
			ret[i][j] = A[j][0] * B[i][j];
		}
	}

	return ret;
}
