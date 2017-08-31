#ifndef __FNN__
#define __FNN__

#include "model.h"
typedef vector<Node*> Layer;

class FNN : public Model
{
private:
	vector<Layer*> layer_list;
public:
	void add_layer();
	void add_input_layer();
	void add_output_layer();

};

#endif