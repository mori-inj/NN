#ifndef __DISCRIMINATOR__
#define __DISCRIMINATOR__

#include "fnn.h"

class Discriminator : public FNN
{
private:
	vector<vector<LD> > grad_X;
public:
	//get gradient on X
	vector<Data> calc_grad_X(Data input_data, function<LD(LD)> deriv_act);
};

#endif