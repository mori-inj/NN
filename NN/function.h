#ifndef __FUNCTION_H__
#define __FUNCTION_H__

#include <math.h>
#define e 2.718281828

//Activation
long double identity(long double x);
long double linear(long double a, long double x);
long double step(long double x);
long double sigmoid(long double x);
long double deriv_sigmoid(long double x);
long double sigmoid(long double b, long double x);
long double deriv_sigmoid(long double b, long double x);
long double ReLU(long double x);

//Cost
long double cross_entropy(long double y, long double h);

#endif