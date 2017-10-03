#ifndef __GDX__
#define __GDX__

#include "fnn.h"

class GDX : public FNN
{
private:
	vector<LD> target_X;
	vector<vector<LD> > grad_X;
public:
	//initialize random X
	void init_X(LD range_min, LD range_max);

	//get gradient on X
	void calc_grad_X(function<LD(LD)> deriv_act);
	
	//GD on X
	void update_grad_X(LD learning_rate);

	//get current X
	vector<LD>& get_X();
};

#endif