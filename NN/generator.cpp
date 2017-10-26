#include "generator.h"
#include "weight.h"
#include "matrix2d.h"

void Generator::dump_weights()
{
	weights.clear();
	int l;
	for(l=0; l<(int)layer_list.size(); l++) {
		int row = (l==0) ? (int)input_layer->size() : (int)layer_list[l-1]->size();
		int col = (int)layer_list[l]->size();
		weights.push_back(new Matrix2D<Weight*>(row, col));
		Layer layer = *layer_list[l];
		for(int i=0; i<col; i++) {
			Node node = *layer[i];
			for(int j=0; j<row; j++) {
				(*weights[l])[j][i] = node.input_weight_list[j];
			}
		}
	}

	int row = (int)layer_list[l-1]->size();
	int col = (int)output_layer->size();
	weights.push_back(new Matrix2D<Weight*>(row, col));
	Layer layer = *output_layer;
	for(int i=0; i<col; i++) {
		Node node = *layer[i];
		for(int j=0; j<row; j++) {
			(*weights[l])[j][i] = node.input_weight_list[j];
		}
	}
}

void Generator::train(long double learning_rate, Data& random_data, vector<Data>& grad_D, LD output_D)
{
	vector<Matrix2D<LD> > g_prime;
	vector<Matrix2D<LD> > theta;
	vector<Matrix2D<LD> > g_prime_theta;
	vector<Matrix2D<LD> > accumulative_g_prime_theta;
	vector<Matrix2D<LD> > grad;
	const int LAYER_SIZE = (int)layer_list.size();

	printf("G train start\n"); fflush(stdout);

	g_prime.push_back(get_deriv_output(random_data));
	for(int l=LAYER_SIZE-1; l>=0; l--) {
		g_prime.push_back(get_layer_deriv_output(l, random_data));
	}
	printf("\tg' (%d) done\n", (int)g_prime.size()); fflush(stdout);
	for(int i=0; i<(int)g_prime.size(); i++) {
		printf("\t\t%3d: %d %d\n", i, g_prime[i].row, g_prime[i].col); fflush(stdout);
		for(int j=0; j<g_prime[i].row; j++) {
			printf("\t\t");
			for(int k=0; k<g_prime[i].col; k++) {
				printf("%.2Lf ", g_prime[i][j][k]);
			}
			printf("\n"); fflush(stdout);
		}
	}

	for(int l=(int)weights.size()-1; l>=0; l--) {
		theta.push_back(Matrix2D<LD>(weights[l]->row, weights[l]->col));
		for(int i=0; i<weights[l]->row; i++) {
			for(int j=0; j<weights[l]->col; j++) {
				theta[(int)weights.size()-1 - l][i][j] = (*weights[l])[i][j]->get_w();
			}
		}
	}
	printf("\ttheta (%d) done\n", (int)theta.size()); fflush(stdout);
	for(int i=0; i<(int)theta.size(); i++) {
		printf("\t\t%3d: %d %d\n", i, theta[i].row, theta[i].col); fflush(stdout);
		for(int j=0; j<theta[i].row; j++) {
			printf("\t\t");
			for(int k=0; k<theta[i].col; k++) {
				printf("%.2Lf ", theta[i][j][k]);
			}
			printf("\n"); fflush(stdout);
		}
	}


	for(int l=0; l<=LAYER_SIZE; l++) {
		g_prime_theta.push_back(mul(g_prime[l], theta[l]));
	}
	printf("\tg' * theta (%d) done\n", (int)g_prime_theta.size()); fflush(stdout);
	for(int i=0; i<(int)g_prime_theta.size(); i++) {
		printf("\t\t%3d: %d %d\n", i, g_prime_theta[i].row, g_prime_theta[i].col); fflush(stdout);
		for(int j=0; j<g_prime_theta[i].row; j++) {
			printf("\t\t");
			for(int k=0; k<g_prime_theta[i].col; k++) {
				printf("%.2Lf ", g_prime_theta[i][j][k]);
			}
			printf("\n"); fflush(stdout);
		}
	}
	

	accumulative_g_prime_theta.push_back(g_prime_theta[0]);
	for(int l=0; l<LAYER_SIZE; l++) {
		accumulative_g_prime_theta.push_back(mat_mul(g_prime_theta[l+1], accumulative_g_prime_theta[l]));
	}
	printf("\taccm g' * theta (%d) done\n", (int)accumulative_g_prime_theta.size()); fflush(stdout);
	for(int i=0; i<(int)accumulative_g_prime_theta.size(); i++) {
		printf("\t\t%3d: %d %d\n", i, accumulative_g_prime_theta[i].row, accumulative_g_prime_theta[i].col); fflush(stdout);
		for(int j=0; j<accumulative_g_prime_theta[i].row; j++) {
			printf("\t\t");
			for(int k=0; k<accumulative_g_prime_theta[i].col; k++) {
				printf("%.2Lf ", accumulative_g_prime_theta[i][j][k]);
			}
			printf("\n"); fflush(stdout);
		}
	}



	for(int l=0; l<=LAYER_SIZE; l++) {
		printf("layer %d (%d x %d): \n", l, 
			accumulative_g_prime_theta[l].row, accumulative_g_prime_theta[l].col); fflush(stdout);
		for(int i=0; i<accumulative_g_prime_theta[l].row; i++) {
			for(int j=0; j<accumulative_g_prime_theta[l].col; j++) {
				printf("%.2Lf ", accumulative_g_prime_theta[l][i][j]); fflush(stdout);
			}
			printf("\n"); fflush(stdout);
		}
		printf("\n\n");
	}
	printf("G train end\n"); fflush(stdout);
	/*

	//여기 아래부터 새로 구하기
	Data deriv_g = get_deriv_output(random_data);
	Data out = get_output(random_data);
	//grad.push_back(Matrix2D<LD>((int)deriv_g.size(), (int)deriv_g.size()*(int)out.size()));
	for(int i=0; i<grad[0].row; i++) {
		for(int j1=0; j1<grad[0].col; j1++) {
			for(int j2=0; j2<grad[0].row; j2++) {

			}
		}
	}


	for(int l=LAYER_SIZE-1; l>=0; l--) {
		//모든 레이어에 대해 grad구하기
	}

	//맞나 검증하고 log씌워서 생긴거 넣고 실제 update 하는것도 넣고

	*/
}