#ifndef __NODE__
#define __NODE__

#include <vector>
#include <functional>

#define LD long double

class Weight;

using namespace std;

class Node
{
private:
	long double input;
	long double target_output;
	long double bias;
	int idx;
	long double delta;
	long double cached_output;
	long double cached_linear_output;
	vector<long double> grad;
public:
	bool is_output_cached;
	bool is_linear_output_cached;

	bool is_input_node;
	bool is_output_node;

	function<long double(long double)> activation_function;
	function<long double(long double)> deriv_activation_function;

	vector<Weight*> input_weight_list;
	vector<Weight*> output_weight_list;

	Node(){};
	Node(int idx, function<LD(LD)> act, function<LD(LD)> deriv_act);
	Node(Node* node, int idx);
	void set_idx(int idx);
	int get_idx();
	void set_bias(long double b);
	long double get_bias();
	long double get_input();
	void set_input(long double input);
	long double get_linear_output();
	long double get_output();
	void set_target_output(long double x);
	long double get_delta();
	long double calc_delta();
	void calc_grad();
	void update_weight(long double learning_rate);
};

#endif
