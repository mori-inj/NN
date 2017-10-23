#include "generator.h"
#include "weight.h"

void Generator::dump_weights()
{
	weights = vector<vector<vector<Weight*> > >();
	int l;
	for(l=0; l<(int)layer_list.size(); l++) {
		weights.push_back(vector<vector<Weight*> >());
		Layer layer = *layer_list[l];
		for(int i=0; i<(int)layer.size(); i++) {
			weights[l].push_back(vector<Weight*>());
			Node node = *layer[i];
			for(int j=0; j<(int)node.input_weight_list.size(); j++) {
				weights[l][i].push_back(node.input_weight_list[j]);
			}
		}
	}

	weights.push_back(vector<vector<Weight*> >());
	Layer layer = *output_layer;
	for(int i=0; i<(int)layer.size(); i++) {
		weights[l].push_back(vector<Weight*>());
		Node node = *layer[i];
		for(int j=0; j<(int)node.input_weight_list.size(); j++) {
			weights[l][i].push_back(node.input_weight_list[j]);
		}
	}
}

void Generator::train(long double learning_rate, Data& random_data, vector<Data>& grad_D, LD output_D)
{
	vector<vector<vector<LD> > > deriv_g_cross_theta;
	vector<vector<vector<LD> > > accumulative_deriv_g_cross_theta;
	vector<vector<vector<LD> > > grad;

	const int LAYER_NUM = (int)layer_list.size();
	for(int l=0; l<=LAYER_NUM; l++) {
		deriv_g_cross_theta.push_back(vector<vector<LD> >());

		//g'(th_(k-1) * a_(k-1))
		Data deriv_g = (l==0) ? get_deriv_output(random_data) 
								: get_layer_deriv_output(LAYER_NUM - l, random_data);

		//th_(k)
		vector<vector<Weight*> >& theta = weights[LAYER_NUM - l];

		int row = (l==0) ? (int)output_layer->size()
						: (int)layer_list[LAYER_NUM - l]->size();
		int column = (l==LAYER_NUM) ? (int)input_layer->size()
									: (int)layer_list[LAYER_NUM - l - 1]->size();

		for(int i=0; i<row; i++) {
			deriv_g_cross_theta[l].push_back(vector<LD>());
			for(int j=0; j<column; j++) {
				LD w = theta[i][j] -> get_w();
				w *= deriv_g[j];
				deriv_g_cross_theta[l][i].push_back(w);
			}
		}
	}

	accumulative_deriv_g_cross_theta.push_back(vector<vector<LD> >());
	accumulative_deriv_g_cross_theta[0] = deriv_g_cross_theta[0];
	for(int l=1; l<=LAYER_NUM-1; l++) {
		accumulative_deriv_g_cross_theta.push_back(vector<vector<LD> >());
		for(int i=0; i<(int)accumulative_deriv_g_cross_theta[l-1].size(); i++) {
			accumulative_deriv_g_cross_theta[l].push_back(vector<LD>());
			for(int j=0; j<(int)deriv_g_cross_theta[l][0].size(); j++) {
				LD sum = 0;
				for(int k=0; k<(int)accumulative_deriv_g_cross_theta[l-1][0].size(); k++) {
					sum += accumulative_deriv_g_cross_theta[l-1][i][k] * deriv_g_cross_theta[l][k][j];
				}
				accumulative_deriv_g_cross_theta[l][i].push_back(sum);
			}
		}
	}



}