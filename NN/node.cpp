#include "node.h"
#include "weight.h"
#include "function.h"

extern int OUTPUT_CNT;

Node::Node(int idx, function<LD(LD)> act, function<LD(LD)> deriv_act)
{
	bias = 0.5;
	this->idx = idx;
	is_input_node = false;
	is_output_node = false;
	
	is_output_cached = false;
	is_linear_output_cached = false;
	
	target_output = 0;

	activation_function = act;
	deriv_activation_function = deriv_act;
}

Node::Node(Node* node, int idx)
{
	this->idx = idx;
	input = node->get_input();
	bias = node->get_bias();
	
	is_input_node = node->is_input_node;
	is_output_node = node->is_output_node;

	is_output_cached = node->is_output_cached;
	is_linear_output_cached = node->is_linear_output_cached;
	
	target_output = 0;

	activation_function = node->activation_function;
	deriv_activation_function = node->deriv_activation_function;
}

void Node::set_idx(int idx)
{
	this->idx = idx;
}

int Node::get_idx()
{
	return idx;
}

long double Node::get_bias()
{
	return bias;
}

long double Node::get_input()
{
	return input;
}

void Node::set_input(long double input)
{
	this->input = input;
}

long double Node::get_linear_output()
{
	long double sum = 0;
	if(is_input_node)
		return this->input;

	if(is_linear_output_cached)
		return cached_linear_output;

	const int WEIGHT_SIZE = (int)input_weight_list.size();
	for(int i=0; i<WEIGHT_SIZE; i++)
	{
		sum += ( input_weight_list[i]->get_src()->get_output() ) * ( input_weight_list[i]->get_w() );
	}
	sum += bias;

	return cached_linear_output = sum;
}

long double Node::get_output() //TODO: needs to be cached
{	
	OUTPUT_CNT++;

	if(is_output_cached)
		return cached_output;

	is_output_cached = true;
	long double sum = get_linear_output();
	if(is_input_node) {
		return cached_output = sum;
	}
	return cached_output = activation_function(sum);
}

void Node::set_target_output(long double x)
{
	target_output = x;
}

long double Node::get_delta()
{
	return delta;
}

long double Node::calc_delta() //does not check get_delta is valid
{
	if(output_weight_list.empty()) {
		delta = get_output() - target_output;
	} else {
		delta = 0;
		for(auto it : output_weight_list) {
			delta += it->get_w() * it->get_dst()->calc_delta();
		}
		delta *= deriv_activation_function(get_linear_output());
	}
	return delta;
}

void Node::calc_grad()
{
	grad.clear();

	long double output = get_output();
	for(auto it : output_weight_list) {
		grad.push_back(output * it->get_dst()->get_delta());
	}

	/*for(auto it : input_weight_list) {
		grad.push_back(it->getSrc()->get_output() * delta);
	}*/
}

void Node::update_weight(long double learning_rate)
{
	for(int i=0; i<(int)output_weight_list.size(); i++) {
		Weight* w = output_weight_list[i];
		long double weight = w->get_w();
		weight = weight - learning_rate * grad[i];
		w->set_w(weight);
	}
	bias -= learning_rate * delta;
}
