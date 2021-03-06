#ifndef __MODEL__
#define __MODEL__

#include <vector>
#include <set>
#include <mutex>
#include "node.h"

typedef vector<long double> Data; // change it to specific class if needed later
typedef int Idx;

using namespace std;

class Model
{
protected:
	mutex get_error_lock, get_precision_lock;

	vector<Node*> input_node_list;
	vector<Node*> output_node_list;

	vector<Node*> node_list;
	set<pair<Node*, Node*> > weight_set;
public:
	Model();
	~Model();
	int get_node_num() { return (int)node_list.size(); }
	
	void add_new_node(function<LD(LD)> act, function<LD(LD)> deriv_act);
	void add_new_input_node();
	void add_new_output_node();
	void add_node(Node* node);
	Node* get_node_by_idx(int idx);
	Idx get_idx_by_node(Node* node);
	void remove_node(Node* node);
	void reindex();
	
	void add_weight(Idx start_node_idx, Idx end_node_idx);
	void add_weight(Idx start_node_idx, Idx end_node_idx, long double w);
	void add_weights(vector<Idx> start_node_idx_list, vector<Idx> end_node_idx_list);
	bool check_weight_exists(Node* a, Node* b);
	void update_weight_set(Weight* w);
	void remove_weight(Weight* w);
	void remove_weight(Idx start_node_idx, Idx end_node_idx);
	
	vector<Node*>::iterator get_first_node_iter();
	vector<Node*>::iterator get_last_node_iter();
	
	void train(long double learning_rate, Data& input_data, Data& output_data);
	void train(long double learning_rate, vector<Data>& input_data_list, vector<Data>& output_data_list);
	Data get_output(Data& input_data);
	Data get_linear_output(Data& input_data);
	vector<Data> get_output(vector<Data>& input_data_list);
	long double cross_entropy_multi(Data& y, Data& h);
	long double get_error(Data& input_data, Data& output_data);
	long double get_error(vector<Data>& input_data_list, vector<Data>& output_data_list);
	void get_error_per_thread(vector<Data>& input_data_list, vector<Data>& output_data_list, pair<int, int> range, LD* error_sum);
	long double get_error_prll(vector<Data>& input_data_list, vector<Data>& output_data_list, int NUM_THREAD);
	long double get_precision(Data& input_data, Data& output_data);
	long double get_precision(vector<Data>& input_data_list, vector<Data>& output_data_list);
	long double get_precision_all(vector<Data>& input_data_list, vector<Data>& output_data_list);
	void get_precision_per_thread(vector<Data>& input_data_list, vector<Data>& output_data_list, pair<int, int> range, LD* precision_sum);
	long double get_precision_prll(vector<Data>& input_data_list, vector<Data>& output_data_list, int NUM_THREAD);
	
	int count_input_node();

	void print();
	void print_bias_and_weights();
	void read_bias_and_weights(char* filename);
};

#endif
