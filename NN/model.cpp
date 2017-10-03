#include "model.h"
#include "weight.h"
#include "function.h"
#include <assert.h>

Model::Model()
{
}

Model::~Model()
{
	const int NODE_LIST_SIZE = (int)node_list.size();
	for(int i=0; i<NODE_LIST_SIZE; i++) {
		Node* node = node_list[i];
		delete(node);
	}
}



void Model::add_new_node(function<LD(LD)> act, function<LD(LD)> deriv_act)
{
	Node* node = new Node((int)node_list.size(), act, deriv_act);
	add_node(node);
}

void Model::add_new_input_node()
{
	Node* node = new Node((int)node_list.size(), 
		//this doesn't need to be done for input nodes
		[](LD x) -> LD{return idt(x);},
		[](LD x) -> LD{return 1;}
	);
	node->is_input_node = true;
	add_node(node);
	input_node_list.push_back(node);
}

void Model::add_new_output_node()
{
	Node* node = new Node((int)node_list.size(),
		[](LD x) -> LD{return sigmoid(x);},
		[](LD x) -> LD{return deriv_sigmoid(x);}
	);
	node->is_output_node = true;
	add_node(node);
	output_node_list.push_back(node);
}

void Model::add_node(Node* node)
{
	node_list.push_back(node);
}

Node* Model::get_node_by_idx(int idx)
{
	return node_list[idx];
}

Idx Model::get_idx_by_node(Node* node)
{
	const int NODE_LIST_SIZE = (int)node_list.size();
	for(int i=0; i<NODE_LIST_SIZE; i++) {
		if(node_list[i] == node) {
			return i;
		}
	}

	return -1;
}

void Model::remove_node(Node* node)
{
	int node_list_size = (int)node_list.size();
	auto node_it = find(node_list.begin(), node_list.end(), node);
	if(node_it != node_list.end()) {
		node_list.erase(node_it);
	}

	if(node->is_input_node) {
		int node_list_size = (int)input_node_list.size();
		auto node_it = find(input_node_list.begin(), input_node_list.end(), node);
		if(node_it != input_node_list.end()) {
			input_node_list.erase(node_it);
		}
	}

	if(node->is_output_node) {
		int node_list_size = (int)output_node_list.size();
		auto node_it = find(output_node_list.begin(), output_node_list.end(), node);
		if(node_it != output_node_list.end()) {
			output_node_list.erase(node_it);
		}
	}

	for(auto pair : weight_set) {
		if(pair.first == node || pair.second == node) {
			weight_set.erase(pair);
		}
	}
	
	delete(node);
	reindex();
}

void Model::reindex()
{
	const int NODE_LIST_SIZE = (int)node_list.size();
	for(int i=0; i<NODE_LIST_SIZE; i++){
		node_list[i]->set_idx(i);
	}
}



void Model::add_weight(Idx start_node_idx, Idx end_node_idx)
{
	Node* start_node = get_node_by_idx(start_node_idx);
	Node* end_node = get_node_by_idx(end_node_idx);

	if(check_weight_exists(start_node, end_node)) {
		return;
	}

	Weight* weight = new Weight(start_node, end_node);
	start_node -> output_weight_list.push_back(weight);
	end_node -> input_weight_list.push_back(weight);
	
	update_weight_set(weight);
}

void Model::add_weights(vector<Idx> start_node_idx_list, vector<Idx> end_node_idx_list)
{
	for(auto i : start_node_idx_list) {
		for(auto j : end_node_idx_list) {
			add_weight(i,j);
		}
	}
}

bool Model::check_weight_exists(Node* a, Node* b)
{
	if(a == b) {
		return true;
	}
	if(weight_set.find(make_pair(a,b)) != weight_set.end()) {
		return true;
	}
	return false;
}

void Model::update_weight_set(Weight* w)
{
	weight_set.insert(make_pair(w->get_src(),w->get_dst()));
}

void Model::remove_weight(Weight* w)
{
	Idx start_idx = get_idx_by_node(w -> get_src());
	Idx end_idx = get_idx_by_node(w -> get_dst());
	
	remove_weight(start_idx, end_idx);
}

void Model::remove_weight(Idx start_node_idx, Idx end_node_idx)
{
	Node* start_node = get_node_by_idx(start_node_idx);
	Node* end_node = get_node_by_idx(end_node_idx);
	
	vector<Weight*>& start_list = start_node -> output_weight_list;
	const int START_LIST_SIZE = (int)start_list.size();
	for(int i=0; i<START_LIST_SIZE; i++) {
		if(start_list[i] -> get_dst() == end_node) {
			start_list.erase(start_list.begin() + i);
		}
	}

	vector<Weight*>& end_list = end_node -> input_weight_list;
	const int END_LIST_SIZE = (int)end_list.size();
	for(int i=0; i<END_LIST_SIZE; i++) {
		if(end_list[i] -> get_src() == start_node) {
			end_list.erase(end_list.begin() + i);
		}
	}

	weight_set.erase(make_pair(start_node, end_node));
}



vector<Node*>::iterator Model::get_first_node_iter()
{
	return node_list.begin();
}

vector<Node*>::iterator Model::get_last_node_iter()
{
	return node_list.end();
}



void Model::train(long double learning_rate, int ITER, Data& input_data, Data& output_data)
{
	assert(input_data.size() == input_node_list.size());
	assert(output_data.size() == output_node_list.size());

	const int INPUT_SIZE = (int)input_data.size();
	const int OUTPUT_SIZE = (int)output_data.size();

	//input node는 input 설정
	for(int i = 0; i < INPUT_SIZE; i++) {
		input_node_list[i] -> set_input(input_data[i]);
	}

	//output node는 output 설정
	for(int i = 0; i < OUTPUT_SIZE; i++) {
		output_node_list[i] -> set_target_output(output_data[i]);
	}

	for(int iter = 0; iter<ITER; iter++) {
		//cache clear
		for(auto node : node_list) {
			node -> is_output_cached = false;
			node -> is_linear_output_cached = false;
		}

		//delta, grad 계산
		//input node자체는 delta구할 필요 없지만 재귀적으로 구하기 위해 input부터 시작
		for(auto node : input_node_list) {
			node -> calc_delta();
		}
		//output node는 grad필요 없음
		for(auto node : node_list) {
			node -> calc_grad();
		}

		//가중치 갱신
		for(auto node : node_list) {
			node -> update_weight(learning_rate);
		}
	}
}

void Model::train(long double learning_rate, int ITER, vector<Data>& input_data_list, vector<Data>& output_data_list)
{
	assert(input_data_list.size() == output_data_list.size());

	const int OUTPUT_DATA_LIST_SIZE = (int)output_data_list.size();
	const int BATCH_SIZE = 100;
	for(int i=rand()%BATCH_SIZE; i<OUTPUT_DATA_LIST_SIZE; i+=BATCH_SIZE) {
		printf("\t%d/%d\n", i, OUTPUT_DATA_LIST_SIZE); fflush(stdout);
		train(learning_rate, ITER, input_data_list[i], output_data_list[i]);
	}
}

Data Model::get_output(Data& input_data)
{
	Data output;
	const int INPUT_DATA_SIZE = (int)input_data.size();
	
	for(auto node : node_list) {
			node -> is_output_cached = false;
			node -> is_linear_output_cached = false;
	}

	for(int i=0; i<INPUT_DATA_SIZE; i++) {
		input_node_list[i] -> set_input(input_data[i]);
	}

	for(auto node : output_node_list) {
		output.push_back(node -> get_output());
	}

	return output;
}

vector<Data> Model::get_output(vector<Data>& input_data_list)
{
	vector<Data> output_list;
	const int INPUT_DATA_LIST_SIZE = (int)input_data_list.size();
	for(int i=0; i<INPUT_DATA_LIST_SIZE; i++) {
		output_list.push_back(get_output(input_data_list[i]));
	}
	return output_list;
}

long double Model::cross_entropy_multi(Data& y, Data& h)
{
	assert(y.size() == h.size());
	int SIZE = (int)y.size();
	long double error_sum = 0;
	for(int i=0; i<SIZE; i++) {
		error_sum += cross_entropy(y[i], h[i]);
	}

	return error_sum;
}

long double Model::get_error(Data& input_data, Data& output_data)
{
	Data output = get_output(input_data);

	return cross_entropy_multi(output_data, output);
}

long double Model::get_error(vector<Data>& input_data_list, vector<Data>& output_data_list)
{
	assert(input_data_list.size() == output_data_list.size());
	long double error_sum = 0;
	int OUTPUT_SIZE = (int)output_data_list.size();
	
	for(int i=0; i<OUTPUT_SIZE; i++) {
		error_sum += get_error(input_data_list[i], output_data_list[i]);
	}

	return error_sum / OUTPUT_SIZE;
}

long double Model::get_precision(Data& input_data, Data& output_data)
{
	int SIZE = (int)output_data.size();
	long double precision = 0;
	Data output = get_output(input_data);

	for(int i=0; i<SIZE; i++) {
		if(output_data[i] == 1)
			precision += output[i];
		else
			precision += 1-output[i];
	}
	
	return precision;
}

long double Model::get_precision(vector<Data>& input_data_list, vector<Data>& output_data_list)
{
	assert(input_data_list.size() == output_data_list.size());
	long double precision = 0;
	int OUTPUT_SIZE = (int)output_data_list.size();

	for(int i=0; i<OUTPUT_SIZE; i++) {
		precision += get_precision(input_data_list[i], output_data_list[i]);
	}

	return precision / OUTPUT_SIZE;
}



int Model::count_input_node()
{
	return (int)input_node_list.size();
}



void Model::print()
{
	printf("===========\nnode list:\n");
	for(int i=0; i<(int)node_list.size(); i++)
	{
		printf("%d\n",i);
		for(int j=0; j<(int)node_list[i]->input_weight_list.size(); j++) {
			printf("  %d -> %d\n", node_list[i]->input_weight_list[j]->get_src()->get_idx(), i); fflush(stdout);
		}
		for(int j=0; j<(int)node_list[i]->output_weight_list.size(); j++) {
			printf("  %d -> %d\n", i, node_list[i]->output_weight_list[j]->get_dst()->get_idx()); fflush(stdout);
		}
	}
	printf("============\n\n"); fflush(stdout);

	printf("===========\nweight set:\n");
	for(auto w : weight_set) {
		printf("%d -> %d\n",w.first->get_idx(),w.second->get_idx()); fflush(stdout);
	}
	printf("============\n\n"); fflush(stdout);
}

void Model::print_weights()
{
	const int NODE_NUM = (int)node_list.size();
	for(int i=0; i<NODE_NUM; i++) {
		printf("%d:%Lf\n",i,node_list[i]->get_bias());
	}
	for(int i=0; i<NODE_NUM; i++) {
		const int OUTPUT_WEIGHT_NUM = (int)node_list[i]->output_weight_list.size();
		for(int j=0; j<OUTPUT_WEIGHT_NUM; j++) {
			printf("%d->%d:%Lf\n", 
				i, 
				node_list[i]->output_weight_list[j]->get_dst()->get_idx(), 
				node_list[i]->output_weight_list[j]->get_w()
				);
			fflush(stdout);
		}
	}
}