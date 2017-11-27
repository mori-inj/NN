#include "fnn.h"

void FNN::add_layer(int num, function<LD(LD)> act, function<LD(LD)> deriv_act)
{
	Layer* layer = new Layer();
	for(int i=0; i<num; i++) {
		Node* node = new Node((int)node_list.size(), act, deriv_act);
		add_node(node);
		layer->push_back(node);
	}

	layer_list.push_back(layer);
}

void FNN::add_input_layer(int num)
{
	input_layer = &input_node_list;
	for(int i=0; i<num; i++) {
		add_new_input_node();
	}
}

void FNN::add_output_layer(int num)
{
	output_layer = &output_node_list;
	for(int i=0; i<num; i++) {
		add_new_output_node();
	}
}

void FNN::add_all_weights()
{
	const int INPUT_LAYER_SIZE = (int)input_layer->size();
	const int OUTPUT_LAYER_SIZE = (int)output_layer->size();

	const int LAYER_NUM = (int)layer_list.size();
	const int FIRST_LAYER_SIZE = (int)layer_list[0]->size();
	const int LAST_LAYER_SIZE = (int)layer_list[LAYER_NUM-1]->size();

	for(int i=0; i<INPUT_LAYER_SIZE; i++) {
		for(int j=0; j<FIRST_LAYER_SIZE; j++) {
			int start = (*input_layer)[i]->get_idx();
			int end = (*layer_list[0])[j]->get_idx();
			add_weight(start, end);
		}
	}

	for(int i=0; i<OUTPUT_LAYER_SIZE; i++) {
		for(int j=0; j<LAST_LAYER_SIZE; j++) {
			int end = (*output_layer)[i]->get_idx();
			int start = (*layer_list[LAYER_NUM-1])[j]->get_idx();
			add_weight(start, end);
		}
	}

	for(int i=0; i<LAYER_NUM-1; i++) {
		int CURR_SIZE = (int)layer_list[i]->size();
		int NEXT_SIZE = (int)layer_list[i+1]->size();
		for(int j=0; j<CURR_SIZE; j++) {
			for(int k=0; k<NEXT_SIZE; k++) {
				int start = (*layer_list[i])[j]->get_idx();
				int end = (*layer_list[i+1])[k]->get_idx();
				add_weight(start, end);
			}
		}
	}
}

int FNN::get_input_size()
{
	return (int)input_layer->size();
}

int FNN::get_output_size()
{
	return (int)output_layer->size();
}


Data FNN::get_layer_output(int l, Data& input_data)
{
	if(l == (int)layer_list.size()) return get_output(input_data);

	get_output(input_data);
	Data ret;
	for(int i=0; i<(int)layer_list[l]->size(); i++) {
		ret.push_back((*layer_list[l])[i]->get_output());
	}
	return ret;
}

Data FNN::get_layer_deriv_output(int l, Data& input_data)
{
	if(l == (int)layer_list.size()) return get_deriv_output(input_data);

	get_output(input_data);
	Data ret;
	for(int i=0; i<(int)layer_list[l]->size(); i++) {
		Node* node = (*layer_list[l])[i];
		ret.push_back(node->deriv_activation_function(node->get_linear_output()));
	}
	return ret;
}

Data FNN::get_layer_linear_output(int l, Data& input_data)
{
	get_output(input_data);
	Data ret;
	for(int i=0; i<(int)layer_list[l]->size(); i++) {
		Node* node = (*layer_list[l])[i];
		ret.push_back(node->get_linear_output());
	}
	return ret;
}


Data FNN::get_deriv_output(Data& input_data)
{
	Data ret = get_linear_output(input_data);
	for(int i=0; i<(int)output_layer->size(); i++) {
		Node* node = (*output_layer)[i];
		ret[i] = node->deriv_activation_function(ret[i]);
	}

	return ret;
}