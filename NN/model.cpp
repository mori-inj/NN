#include "model.h"
#include "weight.h"
#include "function.h"
#include <assert.h>

Model::Model()
{
}

Model::~Model()
{
	for(int i=0; i<(int)node_list.size(); i++) {
		Node* node = node_list[i];
		delete(node);
	}
	for(int i=0; i<(int)weight_list.size(); i++) {
		Weight* weight = weight_list[i];
		delete(weight);
	}
}



void Model::add_new_node()
{
	//TODO	
}

void Model::add_node(Node* node)
{
	node_list.push_back(node);
	printf("===========\nnode list:\n");
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
	printf("============\n\n"); fflush(stdout);
}

Node* Model::get_node_by_idx(int idx)
{
	return node_list[idx];
}



void Model::add_weight(Idx start_node_idx, Idx end_node_idx)
{
	Node* start_node = get_node_by_idx(start_node_idx);
	Node* end_node = get_node_by_idx(end_node_idx);

	if(check_weight_exists(start_node, end_node)) {
		return;
	}

	//TODO
}

void Model::add_weights(vector<Idx> start_node_idx_list, vecrot<Idx> end_node_idx_list)
{
	//TODO
}

bool Model::check_weight_exists(Node* a, Node* b)
{
	if(a == b)
		return true;
	if(weight_set.find(make_pair(a,b)) != weight_set.end())
		return true;
	return false;
}

void Model::update_weight_set(Weight* w)
{
	weight_set.insert(make_pair(w->getSrc(),w->getDst()));
	weight_list.push_back(w);
}

void Model::remove_weight_set(Weight* w)
{
	weight_set.erase(make_pair(w->getSrc(),w->getDst()));
}



vector<Node*>::iterator Model::get_first_node_iter()
{
	return node_list.begin();
}

vector<Node*>::iterator Model::get_last_node_iter()
{
	return node_list.end();
}

void Model::erase_node(Node* a)
{
	for(int i=0; i<(int)node_list.size(); i++){
		if(node_list[i] == a) {
			node_list.erase(node_list.begin() + i);
			break;
		}
	}
	delete(a);
}

void Model::reindex()
{
	for(int i=0; i<(int)node_list.size(); i++){
		node_list[i]->set_idx(i);
	}
}



void Model::train(long double learning_rate, Data& input_data, Data& output_data)
{
	assert(input_data.size() == input_node_list.size());
	assert(output_data.size() == output_node_list.size());

	const int INPUT_SIZE = (int)input_data.size();
	const int OUTPUT_SIZE = (int)output_data.size();

	for(int iter = 0; iter<100; iter++) {
		//�̰� ��ü�� training data��ü�� ���� �ݺ�, �� �� ����ġ ���ſ� ���� �����Ͱ� ������ ������ �ϴ°� �´ٴ°� Ȯ���ϱ�
		for(auto data : input_node_list) {
			
			//input node�� input ����
			for(int i = 0; i < INPUT_SIZE; i++) {
				input_node_list[i] -> set_input(input_data[i]);
			}

			//output node�� output ����
			for(int i = 0; i < OUTPUT_SIZE; i++) {
				output_node_list[i] -> set_target_output(output_data[i]);
			}

			//delta, grad ���
			//input node��ü�� delta���� �ʿ� ������ ��������� ���ϱ� ���� input���� ����
			for(auto node : input_node_list) {
				node -> calc_delta();
			}
			//output node�� grad�ʿ� ����
			for(auto node : node_list) {
				node -> calc_grad();
			}

			//����ġ ���� �� ����plot
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
	for(int i=0; i<(int)output_data_list.size(); i++) {
		train(learning_rate, input_data_list[i], output_data_list[i]);
	}
}

Data Model::get_output(Data& input_data)
{
	Data output;
	for(int i=0; i<(int)input_data.size(); i++) {
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
	for(int i=0; i<(int)input_data_list.size(); i++) {
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
