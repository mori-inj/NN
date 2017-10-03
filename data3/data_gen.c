#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define N 10000

int main(void)
{
	long double x1,x2,tx1,tx2;
	int y;
	int i,j;
	FILE* fp0 = fopen("temp_class0_5000","w");
	FILE* fp1 = fopen("temp_class1_5000","w");

	int cnt0 = 0, cnt1 = 0;

	srand((unsigned)time(NULL));
	for(i=0; 1; i++) {
		tx1 = rand() / (long double) RAND_MAX;
		tx2 = rand() / (long double) RAND_MAX;
		x1 = tx1*2 - 1;
		x2 = tx2*2 - 1;

		if(
				(x1*x1 + x2*x2 > 0.04 && x1*x1 + x2*x2 < 0.09)
				||
				(x1*x1 + x2*x2 > 0.25 && x1*x1 + x2*x2 < 0.36)
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
