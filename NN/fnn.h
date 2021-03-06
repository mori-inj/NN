#ifndef __FNN__
#define __FNN__

#include "model.h"
typedef vector<Node*> Layer;

class FNN : public Model
{
protected:
	Layer* input_layer;
	vector<Layer*> layer_list;
	Layer* output_layer;
public:
	void add_layer(int num, function<LD(LD)> act, function<LD(LD)> deriv_act);
	void add_input_layer(int num);
	void add_output_layer(int num);

	void add_all_weights();

	int get_input_size();
	int get_output_size();

	Data get_layer_output(int l, Data& input_data);
	Data get_layer_deriv_output(int l, Data& input_data);
	Data get_layer_linear_output(int l, Data& input_data);
	
	Data get_deriv_output(Data& input_data);
};

#endif