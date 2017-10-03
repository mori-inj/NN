#include "weight.h"
#include <math.h>

Weight::Weight(Weight& weight)
{
	this->w =  weight.get_w();
	this->src = weight.get_src();
	this->dst = weight.get_dst();
}

Weight::Weight(Node* src)
{
	w = 0;
	while(w==0) {
		w = 2 * ((long double)rand() - RAND_MAX/2) / RAND_MAX;
	}
	this->src = src;
	this->dst = NULL;
}
Weight::Weight(Node* src, Node* dst)
{
	w = 0;
	while(w==0) {
		w = 2 * ((long double)rand() - RAND_MAX/2) / RAND_MAX;
	}
	this->src = src;
	this->dst = dst;
}

Node* Weight::get_src()
{
	return src;
}

Node* Weight::get_dst()
{
	return dst;
}

long double Weight::get_w()
{
	return w;
}

void Weight::set_dst(Node& dst)
{
	this->dst = &dst;
}

void Weight::set_w(long double w)
{
	this->w = w;
}
