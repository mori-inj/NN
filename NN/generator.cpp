#include "generator.h"
#include "weight.h"

void Generator::dump_weights()
{
	weights = vector<Matrix2D<Weight*> >();
	int l;
	for(l=0; l<(int)layer_list.size(); l++) {
		int row = (l==0) ? (int)input_layer->size() : (int)layer_list[l-1]->size();
		int col = (int)layer_list[l]->size();
		weights.push_back(Matrix2D<Weight*>(row, col));
		Layer layer = *layer_list[l];
		for(int i=0; i<(int)layer.size(); i++) {
			Node node = *layer[i];
			for(int j=0; j<(int)node.input_weight_list.size(); j++) {
				weights[l][i][j] = node.input_weight_list[j];
			}
		}
	}

	int row = (int)layer_list[l-1]->size();
	int col = (int)output_layer->size();
	weights.push_back(Matrix2D<Weight*>(row, col));
	Layer layer = *output_layer;
	for(int i=0; i<(int)layer.size(); i++) {
		Node node = *layer[i];
		for(int j=0; j<(int)node.input_weight_list.size(); j++) {
			weights[l][i][j] = node.input_weight_list[j];
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

	g_prime.push_back(get_deriv_output(random_data));
	for(int l=LAYER_SIZE-1; l>=0; l--) {
		g_prime.push_back(get_layer_deriv_output(l, random_data));
	}

	for(int l=(int)weights.size()-1; l>=0; l--) {
		theta.push_back(Matrix2D<LD>(weights[l].row, weights[l].col));
		for(int i=0; i<weights[l].row; i++) {
			for(int j=0; j<weights[l].col; j++) {
				theta[l][i][j] = weights[l][i][j]->get_w();
			}
		}
	}

	for(int l=LAYER_SIZE; l>=0; l--) {
		g_prime_theta.push_back(mul(g_prime[l], theta[l]));
	}

	accumulative_g_prime_theta.push_back(g_prime_theta[0]);
	for(int l=0; l<LAYER_SIZE; l++) {
		accumulative_g_prime_theta.push_back(mat_mul(accumulative_g_prime_theta[l], g_prime_theta[l+1]));
	}

	Data deriv_g = get_deriv_output(random_data);
	Data out = get_output(random_data);
	grad.push_back(Matrix2D<LD>((int)deriv_g.size(), (int)deriv_g.size()*(int)out.size()));
	for(int i=0; i<grad[0].row; i++) {
		for(int j1=0; j1<grad[0].col; j1++) {
			for(int j2=0; j2<grad[0].row; j2++) {
			}
		}
	}


	for(int l=LAYER_SIZE-1; l>=0; l--) {
	}
}