#ifndef __NODE__
#define __NODE__

#include <vector>

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
	vector<long double> grad;
public:
	bool input_node;

	vector<Weight*> input_weight_list;
	vector<Weight*> output_weight_list;

	Node(){};
	Node(int x, int y, int idx);
	Node(Node* node, int idx);
	void set_idx(int idx);
	int get_idx();
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
