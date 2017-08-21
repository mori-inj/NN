#include "model.h"
#include "weight.h"
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
vector<Node*>::iterator Model::get_first_node_iter()
{
	return node_list.begin();
}
vector<Node*>::iterator Model::get_last_node_iter()
{
	return node_list.end();
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

void Model::train(long double learning_rate, set<Node*>& plot_in_list, set<Node*>& plot_out_list, HWND& plotWindowHwnd)
{
	int cnt = 0;
	for(auto node : plot_out_list) {
		if(node->plot_output) {
			cnt++;
		}
	}
	if( (count_input_node() != 2) || (cnt != 1) ) {
		return;
	}

	Node* input[2];

	for(int iter = 0; iter<1; iter++) {
		//이거 전체를 training data전체에 대해 반복, 한 번 가중치 갱신에 쓰인 데이터가 여러번 쓰여야 하는게 맞다는걸 확인하기
		for(auto data : input_data_list) {
			//input node는 input 설정
			input[0] = *(plot_in_list.begin());
			input[1] = *(plot_in_list.rbegin());

			input[0]->set_input(data->getXY().y);
			input[1]->set_input(data->getXY().x);
			//output node는 output 설정
			Node* output_node = NULL;
			for(auto node : plot_out_list) {
				if(node->plot_output) {
					output_node = node;
				}
			}
			output_node -> set_target_output(data->get_class());

			//delta, grad 계산
			//input node자체는 delta구할 필요 없지만 재귀적으로 구하기 위해 input부터 시작
			for(auto node : plot_in_list) {
				node -> calc_delta();
			}
			//output node는 grad필요 없음
			for(auto node : node_list) {
				if(node != output_node) {
					node -> calc_grad();
				}
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

		if(plotWindowHwnd!=NULL)
		{
			SendMessage(plotWindowHwnd, WM_TIMER, NULL, NULL);
		}
	}
}






void Model::setXY_of_idx(int x, int y, int idx)
{
	node_list[idx] -> setXY(x,y);
}
int Model::get_idx_of_clicked_node(int x, int y)
{
	for(int i=0; i<node_list.size(); i++)
	{
		if(node_list[i]->isIn(x, y))
		{
			return node_list[i]->get_idx();
		}
	}
	return -1;
}
bool Model::get_is_selected_left_by_idx(int idx)
{
	return node_list[idx]->is_selected_left;
}
void Model::clear_selected_left()
{
	for(int i = 0; i<(int)node_list.size(); i++)
	{
		node_list[i]->is_selected_left = false;
	}
}
void Model::clear_selected_right()
{
	for(int i = 0; i<(int)node_list.size(); i++)
	{
		node_list[i]->is_selected_right = false;
	}
}
void Model::print(HDC MemDC)
{
	for(auto node : node_list) {
		node -> print_weight(MemDC);
	}
	for(auto node: node_list) {
		node -> print(MemDC);
	}
}

void Model::clear_plot_mode()
{
	for(auto node : node_list) {
		node -> plot_mode = false;
	}
	for(auto w :  weight_list) {
		w -> plot_mode = false;
	}
}

int Model::count_input_node()
{
	int cnt = 0;
	for(int i = 0; i<(int)node_list.size(); i++)
	{
		if(node_list[i]->input_node)
			cnt++;
	}
	return cnt;
}
