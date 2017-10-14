#ifndef __GDX__
#define __GDX__

#include "fnn.h"

class GDX : public FNN
{
private:
	LD MIN, MAX;
	vector<LD> target_X;
	vector<vector<LD> > grad_X;
public:
	int TARGET_CLASS;
	//initialize random X
	void init_X(LD range_min, LD range_max, int target_class);

	//get gradient on X
	void calc_grad_X(function<LD(LD)> deriv_act);
	
	//GD on X
	void update_grad_X(LD learning_rate);

	//get current X
	vector<LD> get_X();

	void set_X(LD range_min, LD range_max, vector<LD> data, int target_class);
};

#endif