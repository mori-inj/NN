#include "function.h"


long double identity(long double x)
{
	return x;
}
long double linear(long double a, long double x)
{
	return a*x;
}

long double step(long double x)
{
	if(x<0)
		return 0;
	else
		return 1;
}

long double sigmoid(long double x)
{
	return 1/(1+powl(e,-x));
}

long double deriv_sigmoid(long double x)
{
	long double s = sigmoid(x);
	return s*(1-s);
}

long double sigmoid(long double b, long double x)
{
	return 1/(1+powl(e,-b*x));
}

long double deriv_sigmoid(long double b, long double x)
{
	long double s = sigmoid(b,x);
	return b*s*(1-s);
}

long double ReLU(long double x)
{
	if(x<=0)
		return 0;
	else
		return x;
}




long double cross_entropy(long double y, long double h)
{
	return - (y * log(h) + (1-y)*log(1-h));
}