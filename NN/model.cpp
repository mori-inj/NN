#include "model.h"
#include "weight.h"
#include "function.h"
#include <assert.h>

using namespace std;

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



void Model::add_new_node()
{
	Node* node = new Node((int)node_list.size());
	add_node(node);
}

void Model::add_node(Node* node)
{
	node_list.push_back(node);
	/*printf("===========\nnode list:\n");
	for(int i=0; i<(int)node_list.size(); i++)
	{
		printf("%d\n",i);
		for(int j=0; j<(int)node_list[i]->input_weight_list.size(); j++) {
			printf("  %d -> %d\n", node_list[i]->input_weight_list[j]->getSrc()->get_idx(), i); fflush(stdout);
		}
		for(int j=0; j<(int)node_list[i]->output_weight_list.size(); j++) {
			printf("  %d -> %d\n", i, node_list[i]->output_weight_list[j]->getDst()->get_idx()); fflush(stdout);
		}
	}
	printf("============\n\n"); fflush(stdout);

	printf("===========\nweight set:\n");
	for(auto w : weight_set) {
		printf("%d -> %d\n",w.first->get_idx(),w.second->get_idx()); fflush(stdout);
	}
	printf("============\n\n"); fflush(stdout);*/
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
	weight_set.insert(make_pair(w->getSrc(),w->getDst()));
}

void Model::remove_weight(Weight* w)
{
	Idx start_idx = get_idx_by_node(w -> getSrc());
	Idx end_idx = get_idx_by_node(w -> getDst());
	
	remove_weight(start_idx, end_idx);
}

void Model::remove_weight(Idx start_node_idx, Idx end_node_idx)
{
	Node* start_node = get_node_by_idx(start_node_idx);
	Node* end_node = get_node_by_idx(end_node_idx);
	
	vector<Weight*>& start_list = start_node -> output_weight_list;
	const int START_LIST_SIZE = (int)start_list.size();
	for(int i=0; i<START_LIST_SIZE; i++) {
		if(start_list[i] -> getDst() == end_node) {
			start_list.erase(start_list.begin() + i);
		}
	}

	vector<Weight*>& end_list = end_node -> input_weight_list;
	const int END_LIST_SIZE = (int)end_list.size();
	for(int i=0; i<END_LIST_SIZE; i++) {
		if(end_list[i] -> getSrc() == start_node) {
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



void Model::train(long double learning_rate, Data& input_data, Data& output_data)
{
	assert(input_data.size() == input_node_list.size());
	assert(output_data.size() == output_node_list.size());

	const int INPUT_SIZE = (int)input_data.size();
	const int OUTPUT_SIZE = (int)output_data.size();

	for(int iter = 0; iter<100; iter++) {
		//이거 전체를 training data전체에 대해 반복, 한 번 가중치 갱신에 쓰인 데이터가 여러번 쓰여야 하는게 맞다는걸 확인하기
		for(auto data : input_node_list) {
			
			//input node는 input 설정
			for(int i = 0; i < INPUT_SIZE; i++) {
				input_node_list[i] -> set_input(input_data[i]);
			}

			//output node는 output 설정
			for(int i = 0; i < OUTPUT_SIZE; i++) {
				output_node_list[i] -> set_target_output(output_data[i]);
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

			//가중치 갱신 및 새로plot
			for(auto node : node_list) {
				node -> update_weight(learning_rate);
			}

			
			/*printf("===========\nnode list:\n");
			for(int i=0; i<(int)node_list.size(); i++)
			{
				printf("%d\n",i);
				for(int j=0; j<(int)node_list[i]->input_weight_list.size(); j++) {
					printf("  %d -> %d  :%Lf\n", node_list[i]->input_weight_list[j]->getSrc()->get_idx(), i, node_list[i]->input_weight_list[j]->getW()); fflush(stdout);
				}
				for(int j=0; j<(int)node_list[i]->output_weight_list.size(); j++) {
					printf("  %d -> %d  :%Lf\n", i, node_list[i]->output_weight_list[j]->getDst()->get_idx(), node_list[i]->output_weight_list[j]->getW()); fflush(stdout);
				}
			}
			printf("============\n\n"); fflush(stdout);*/
			
		}
	}
}

void Model::train(long double learning_rate, vector<Data>& input_data_list, vector<Data>& output_data_list)
{
	assert(input_data_list.size() == output_data_list.size());

	const int OUTPUT_DATA_LIST_SIZE = (int)output_data_list.size();
	for(int i=0; i<OUTPUT_DATA_LIST_SIZE; i++) {
		train(learning_rate, input_data_list[i], output_data_list[i]);
	}
}

Data Model::get_output(Data& input_data)
{
	Data output;
	const int INPUT_DATA_SIZE = (int)input_data.size();
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
	int size = (int)y.size();
	long double error_sum = 0;
	for(int i=0; i<size; i++) {
		error_sum += cross_entropy(y[i], h[i]);
	}

	return error_sum;
}

long double Model::get_error(Data& input_data, Data& output_data)
{
	long double error_sum = 0;
	Data output = get_output(input_data);

	error_sum += cross_entropy_multi(output_data, output);

	return error_sum;
}

long double Model::get_error(vector<Data>& input_data_list, vector<Data>& output_data_list)
{
	assert(input_data_list.size() == output_data_list.size());
	long double error_sum = 0;
	int output_size = (int)output_data_list.size();
	
	for(int i=0; i<output_size; i++) {
		error_sum += get_error(input_data_list[i], output_data_list[i]);
	}

	return error_sum / output_size;
}



int Model::count_input_node()
{
	return (int)input_node_list.size();
}
