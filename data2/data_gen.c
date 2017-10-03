#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define N 1000

int main(void)
{
	long double x1,x2,tx1,tx2;
	int y;
	int i,j;
	FILE* fp0 = fopen("temp_class0_500","w");
	FILE* fp1 = fopen("temp_class1_500","w");

	int cnt0 = 0, cnt1 = 0;

	srand((unsigned)time(NULL));
	for(i=0; 1; i++) {
		tx1 = rand() / (long double) RAND_MAX;
		tx2 = rand() / (long double) RAND_MAX;
		x1 = tx1*2 - 1;
		x2 = tx2*2 - 1;

		if( 
			(
			 ((x1-0.4)*(x1-0.4)+(x2+0.3)*(x2+0.3) > 0.2*0.2) &&
			 ((x1-0.5)*(x1-0.5)+(x2+0.2)*(x2+0.2) < 0.4*0.4)
			)
			||
			((x1+0.7)*(x1+0.7)+(x2-0.5)*(x2-0.5) < 0.3*0.3)
				) {
			y = 1;
			if(cnt1 < N/2) {
				fprintf(fp1, "%lf %lf\n", (double)tx1, (double)tx2);
				cnt1++;
			}
		} else {
			y = 0;
			if(cnt0 < N/2) {
				fprintf(fp0, "%lf %lf\n", (double)tx1, (double)tx2);
				cnt0++;
			}
		}
		if(cnt0 + cnt1 == N) break;
	}

	return 0;
}
