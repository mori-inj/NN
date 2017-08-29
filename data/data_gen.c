#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define N 10000

int main(void)
{
	long double x1,x2;
	int y;
	
	int i,j;
	srand((unsigned)time(NULL));
	for(i=0; i<N; i++) {
		x1 = rand() / (long double) RAND_MAX;
		x2 = rand() / (long double) RAND_MAX;

		if( (x2 < 3*x1 - 0.3) && (x2 > 0.4*x1 + 0.3) && (x2 < -0.2*x1 + 1) ) {
			y = 1;
		} else {
			y = 0;
		}
		if(y==0)
		printf("%Lf %Lf\n",x1, x2); 
	}

	return 0;
}
