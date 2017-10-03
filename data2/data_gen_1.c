#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define N 1000

int main(int argc, char* argv[])
{
	double x1, x2;
	int y;
	
	int i, j, cnt=0, gen_class = argv[1][0]-'0';
	
	srand((unsigned)time(NULL));
	
	for(i=0; cnt<N/2; i++) {
		x1 = rand() / (double) RAND_MAX;
		x2 = rand() / (double) RAND_MAX;

		if( (x2 < 3*x1 - 0.3) && (x2 > 0.4*x1 + 0.3) && (x2 < -0.2*x1 + 1) ) {
			y = 1;
			if(gen_class==1)
				cnt++;
		} else {
			y = 0;
			if(gen_class==0)
				cnt++;
		}
		if(y==gen_class)
			printf("%lf %lf\n",x1, x2); 
	}
	return 0;
}
