#include "node.h"
#include "weight.h"
#include "function.h"

extern int OUTPUT_CNT;

Node::Node(int idx)
{
	bias = 0.5;
	this->idx = idx;
	input_node = ;//true;
	
	target_output = 0;
}

Node::Node(Node* node, int idx)
{
	this->idx = idx;
	input = node->get_input();
	bias = node->get_bias();
	
	input_node = node->input_node;
	
	target_output = 0;
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
	if(input_weight_list.size()==0)
		return this->input;

	for(int i=0; i<(int)input_weight_list.size(); i++)
	{
		sum += ( input_weight_list[i]->getSrc()->get_output() ) * ( input_weight_list[i]->getW() );
	}
	sum += bias;

	return sum;
}

long double Node::get_output() //TODO: needs to be cached
{
	
	OUTPUT_CNT++;
	long double sum = get_linear_output();
	if(input_weight_list.size()==0)
		return sum;

	return sigmoid(10, sum);
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
			delta += it->getW() * it->getDst()->calc_delta();
		}
		delta *= deriv_sigmoid(10, get_linear_output());
	}
	return delta;
}

void Node::calc_grad()
{
	grad.clear();

	long double output = get_output();
	for(auto it : output_weight_list) {
		grad.push_back(output * it->getDst()->get_delta());
	}

	/*for(auto it : input_weight_list) {
		grad.push_back(it->getSrc()->get_output() * delta);
	}*/
}

void Node::update_weight(long double learning_rate)
{
	for(int i=0; i<(int)output_weight_list.size(); i++) {
		Weight* w = output_weight_list[i];
		long double weight = w->getW();
		weight = weight - learning_rate * grad[i];
		w->setW(weight);
	}
	bias -= learning_rate * delta;
}