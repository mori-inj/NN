#ifndef __MODEL__
#define __MODEL__

#include <vector>
#include <set>
#include "node.h"
#include "input_data.h"

using namespace std;

class Model
{
private:
	vector<Node*> node_list;
	set<pair<Node*, Node*> > weight_set;
	vector<Weight*> weight_list;
public:
	vector<InputData*> input_data_list;
	set<InputData*> input_data_set[2];

	Model();
	~Model();
	int get_node_num() { return (int)node_list.size(); }
	void add_node(Node* node);
	Node* get_node_by_idx(int idx);
	bool check_weight_exists(Node* a, Node* b);
	void update_weight_set(Weight* w);
	void remove_weight_set(Weight* w);
	vector<Node*>::iterator get_first_node_iter();
	vector<Node*>::iterator get_last_node_iter();
	void erase_node(Node* node);
	void reindex();
	void train(long double learning_rate, set<Node*>& plot_in_list, set<Node*>& plot_out_list, HWND& plotWindowHwnd);
	int count_input_node()
};

#endif
