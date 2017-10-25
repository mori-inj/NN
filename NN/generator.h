#ifndef __GENERATOR__
#define __GENERATOR__

#include "fnn.h"
#include "matrix2d.h"

class Generator : public FNN
{
private:
	vector<Matrix2D<Weight*> > weights;
public:
	void dump_weights();
	void train(long double learning_rate, Data& random_data, vector<Data>& grad_D, LD output_D);
};

#endif