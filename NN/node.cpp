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
	is_delta_cached = false;
	
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
	is_delta_cached = node->is_delta_cached;

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

void Node::set_bias(long double b)
{
	bias = b;
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
	for(int i=0; i<WEIGHT_SIZE; i++) {
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
	if(is_delta_cached)
		return delta;

	is_delta_cached = true;
	if((int)output_weight_list.size() == 0) {
		delta = get_output() - target_output;
		//for(int i=0; i<lev; i++) printf("\t");
		//printf("%d is output node\n", get_idx()); fflush(stdout);
	} else {
		delta = 0;
		for(auto it : output_weight_list) {
			LD tmp;
			delta += it->get_w() * (tmp=it->get_dst()->calc_delta());
			//for(int i=0; i<lev; i++) printf("\t");
			//printf("%d w:%Lf, dst delta:%Lf\n", get_idx(), it->get_w(),tmp); fflush(stdout);
		}
		delta *= deriv_activation_function(get_linear_output());
	}
	//printf("%d delta: %Lf, output size: %d\n\n", get_idx(), delta, (int)output_weight_list.size()); fflush(stdout);
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
	LD sum = 0;
	for(int i=0; i<(int)output_weight_list.size(); i++) {
		Weight* w = output_weight_list[i];
		long double weight = w->get_w();
		weight = weight - learning_rate * grad[i];
		w->set_w(weight);
		sum += weight*weight;
	}
	bias -= learning_rate * delta;
	sum += bias*bias;

	if(sum>1) {
		for(auto w : output_weight_list) {
			LD weight = w->get_w();
			w->set_w(weight/sum);
		}
		bias /= sum;
	}
}
