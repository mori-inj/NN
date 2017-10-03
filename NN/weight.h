#ifndef __WEIGHT__
#define __WEIGHT__

#include <Windows.h>
#include "node.h"

typedef int NodeIdx;

class Weight
{
protected:
	long double w;
	Node* src;
	Node* dst;
public:
	Weight(){};
	Weight(Node* src);
	Weight(Weight& weight);
	Weight(Node* src, Node* dst);
	Node* get_src();
	Node* get_dst();
	long double get_w();
	void set_dst(Node& dst);
	void set_w(long double w);
};

#endif