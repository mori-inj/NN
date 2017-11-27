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

void Generator::train(long double learning_rate, Data& random_data, vector<Data>& grad_D, Data output_D)
{
	bool DEBUG_MODE = true;//false;
	const int N = (int)layer_list.size()+2;
	vector<Matrix2D<LD> > g_prime(N);
	vector<Matrix2D<LD> > theta(N);
	vector<Matrix2D<LD> > g_prime_theta(N);
	vector<Matrix2D<LD> > accumulative_g_prime_theta(N);
	vector<Matrix2D<LD> > a_prime(N);
	vector<vector<Matrix2D<LD> > > grad_G(N);
	vector<vector<Matrix2D<LD> > > grad(N);
	clock_t t,s;

	t=clock();
	for(int n=1; n<=N-1; n++) {
		g_prime[n] = get_layer_deriv_output(n-1, random_data);
	}
	if(DEBUG_MODE) {
		printf("g' done (%d)\n",clock()-t); fflush(stdout);
	}

	t=clock();
	for(int n=1; n<=N-1; n++) {
		theta[n] = Matrix2D<LD>(weights[n-1]->row, weights[n-1]->col);
		for(int i=0; i<weights[n-1]->row; i++) {
			for(int j=0; j<weights[n-1]->col; j++) {
				theta[n][i][j] = (*weights[n-1])[i][j]->get_w();
			}
		}
	}
	if(DEBUG_MODE) {
		printf("theta done (%d)\n",clock()-t); fflush(stdout);
	}

	t=clock();
	for(int n=1; n<=N-1; n++) {
		g_prime_theta[n] = mul(g_prime[n], theta[n]);
	}
	if(DEBUG_MODE) {
		printf("g' * theta done (%d)\n",clock()-t); fflush(stdout);
	}

	t=clock();
	accumulative_g_prime_theta[N-1] = g_prime_theta[N-1];
	for(int n=N-2; n>=2; n--) {
		accumulative_g_prime_theta[n] = mat_mul(g_prime_theta[n], accumulative_g_prime_theta[n+1]);
	}
	if(DEBUG_MODE) {
		printf("accm g' * theta done (%d)\n",clock()-t); fflush(stdout);
	}

	t=clock();
	for(int n=1; n<=N-1; n++) {
		Matrix2D<LD> g_prime_a;
		Matrix2D<LD> ret;
		if(n==1) {
			ret = Matrix2D<LD>(random_data);
		} else {
			ret = Matrix2D<LD>(get_layer_output(n-2,random_data));
		}
		ret.transpose(ret);
		g_prime_a = mat_mul(g_prime[n], ret);
		g_prime_a.transpose(g_prime_a);
		a_prime[n] = g_prime_a;
	}
	if(DEBUG_MODE) {
		printf("a' done (%d)\n",clock()-t); fflush(stdout);
	}

	t=clock();
	for(int n=1; n<=N-1; n++) {
		const int s_n = get_output_size();
		grad_G[n] = vector<Matrix2D<LD> >(s_n);
		if(n==N-1) {
			for(int i=0; i<s_n; i++) {
				grad_G[n][i] = Matrix2D<LD> (a_prime[n].row,a_prime[n].col);
				for(int j=0; j<a_prime[n].row; j++) {
					for(int k=0; k<a_prime[n].col; k++) {
						if(k==i)
							grad_G[n][i][j][k] = a_prime[n][j][k];
						else
							grad_G[n][i][j][k] = 0;
					}
				}
			}
		} else {
			accumulative_g_prime_theta[n+1].transpose(accumulative_g_prime_theta[n+1]);
			for(int i=0; i<s_n; i++) {
				grad_G[n][i] = Matrix2D<LD> (a_prime[n].row,a_prime[n].col);
				grad_G[n][i] = mul(accumulative_g_prime_theta[n+1][i], a_prime[n]);
			}
		}
	}
	if(DEBUG_MODE) {
		printf("grad_G done (%d)\n",clock()-t); fflush(stdout);
	}

	// grad_D ¶û °öÇÏ±â
	t=clock();
	Matrix2D<LD> D_prime((int)grad_D.size(), (int)grad_D[0].size());
	for(int i=0; i<D_prime.row; i++) {
		D_prime[i] = grad_D[i];
	}
	for(int n=1; n<=N-1; n++) {
		grad[n] = vector<Matrix2D<LD> >(D_prime.row);
		Matrix2D<LD> serialized((int)grad_G[n].size(), 0);
		for(int i=0; i<(int)grad_G[n].size(); i++) {
			for(int j=0; j<grad_G[n][0].row; j++) {
				serialized[i].insert(serialized[i].end(), grad_G[n][i][j].begin(), grad_G[n][i][j].end());
				/*for(int k=0; k<grad_G[n][0].col; k++) {
					serialized[i].push_back(grad_G[n][i][j][k]);
				}*/
			}
		}
		serialized.col = grad_G[n][0].row * grad_G[n][0].col;

		Matrix2D<LD> ret = mat_mul(D_prime, serialized);
		for(int i=0; i<D_prime.row; i++) {
			Matrix2D<LD> deserialized(grad_G[n][0].row,grad_G[n][0].col);
			for(int j=0; j<deserialized.row; j++) {
				deserialized[j] = vector<LD>(ret[i].begin()+j*deserialized.col,ret[i].begin()+(j+1)*deserialized.col);
				/*for(int k=0; k<deserialized.col; k++) {
					deserialized[j][k] = ret[i][j*deserialized.col + k];
				}*/
			}
			grad[n][i] = deserialized;
		}
	}
	if(DEBUG_MODE) {
		printf("grad done (%d)\n",clock()-t); fflush(stdout);
	}

	// ·Î±× ¹ÌºÐ f'/f
	t=clock();
	for(int n=1; n<=N-1; n++) {
		for(int i=0; i<(int)grad[n].size(); i++) {
			const LD ret = max(1e-12, output_D[i]);
			for(int j=0; j<grad[n][i].row; j++) {
				for(int k=0; k<grad[n][i].col; k++) {
					grad[n][i][j][k] /=  ret;
				}
			}
		}
	}
	if(DEBUG_MODE) {
		printf("log done (%d)\n",clock()-t); fflush(stdout);
	}

	//theta update
	t=clock();
	for(int n=1; n<=N-1; n++) {
		for(int k=0; k<(int)grad[n].size(); k++) {
			for(int i=0; i<weights[n-1]->row; i++) {
				for(int j=0; j<weights[n-1]->col; j++) {
					LD w = (*weights[n-1])[i][j]->get_w();
					w = w + learning_rate * grad[n][k][i][j];
					(*weights[n-1])[i][j]->set_w(w);
				}
			}
		}
	}
	if(DEBUG_MODE) {
		printf("theta update done (%d)\n",clock()-t); fflush(stdout);
	}

	//bias update
}