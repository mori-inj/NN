#ifndef __FUNCTION_H__
#define __FUNCTION_H__

#include <math.h>
#define e 2.718281828

//Activation
long double idt(long double x);
long double linear(long double a, long double x);
long double step(long double x);
long double sigmoid(long double x);
long double deriv_sigmoid(long double x);
long double sigmoid(long double b, long double x);
long double deriv_sigmoid(long double b, long double x);
long double ReLU(long double x);
long double deriv_ReLU(long double x);
long double PReLU(long double x);
long double deriv_PReLU(long double x);
long double exponential_converge(long double x);
long double deriv_exponential_converge(long double x);

//Cost
long double cross_entropy(long double y, long double h);

#endif