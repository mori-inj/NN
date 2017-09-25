#include "fnn.h"

void FNN::add_layer(int num)
{
	Layer* layer = new Layer();
	for(int i=0; i<num; i++) {
		Node* node = new Node();
		node_list.push_back(node);
		layer->push_back(node);
		//add weight considering the input/output layers
	}

	//layer_list.push_back(layer);
}

void FNN::add_layer(int num, int idx)
{

}

void FNN::add_input_layer(int num)
{

}

void FNN::add_output_layer(int num)
{

}