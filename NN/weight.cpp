#include "weight.h"
#include <math.h>

Weight::Weight(Weight& weight)
{
	this->w =  weight.getW();
	this->src = weight.getSrc();
	this->dst = weight.getDst();
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

Node* Weight::getSrc()
{
	return src;
}

Node* Weight::getDst()
{
	return dst;
}

long double Weight::getW()
{
	return w;
}

void Weight::setDst(Node& dst)
{
	this->dst = &dst;
}

void Weight::setW(long double w)
{
	this->w = w;
}
