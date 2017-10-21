#include "gdx.h"
#include "weight.h"

//#define PRINT_MODE

//initialize random X
void GDX::init_X(LD range_min, LD range_max, int target_class)
{
	TARGET_CLASS = target_class;
	MIN = range_min;
	MAX = range_max;
	const int INPUT_SIZE = (int)input_layer->size();
	for(int i=0; i<INPUT_SIZE; i++) {
		LD r = rand() / (LD) RAND_MAX;
		r *= (range_max - range_min);
		r += range_min;
		target_X.push_back(r);
	}
}

//get gradient on X
void GDX::calc_grad_X(function<LD(LD)> deriv_act)
{
	vector<vector<LD> > temp_matrix;
	const int INPUT_SIZE = (int)input_layer->size();
	const int OUTPUT_SIZE = (int)output_layer->size();


	//get analytical grad_X for specific class
	/*vector<LD> analytic_grad_X;
	const LD eps = 0.01;
	for(int i = 0; i < INPUT_SIZE; i++) {
		vector<LD> temp_target_X_plus = target_X;
		vector<LD> temp_target_X_minus = target_X;
		temp_target_X_plus[i] += eps;
		temp_target_X_minus[i] -= eps;
		LD delta = get_output(temp_target_X_plus)[TARGET_CLASS] - get_output(temp_target_X_minus)[TARGET_CLASS];
#ifdef PRINT_MODE
		printf("plus: %Lf minus: %Lf\n",get_output(temp_target_X_plus)[TARGET_CLASS], get_output(temp_target_X_minus)[TARGET_CLASS]);
#endif
		analytic_grad_X.push_back(delta/(2*eps));
	}
#ifdef PRINT_MODE
	printf("analytic: ");
	for(int i = 0; i < INPUT_SIZE; i++) {
		printf("%Lf ",analytic_grad_X[i]);
	}
	printf("\n");
	fflush(stdout);
#endif*/

	//get analytical grad_X
	/*vector<Data> analytic_grad_X;
	const LD eps = 0.0001;
	for(int i=0; i<OUTPUT_SIZE; i++) {
		analytic_grad_X.push_back(Data());
		for(int j=0; j<INPUT_SIZE; j++) {
			vector<LD> temp_target_X_plus = target_X;
			vector<LD> temp_target_X_minus = target_X;
			temp_target_X_plus[j] += eps;
			temp_target_X_minus[j] -= eps;
			LD delta = get_output(temp_target_X_plus)[i] - get_output(temp_target_X_minus)[i];
			analytic_grad_X[i].push_back(delta/(2*eps));
		}
	}

	for(int i=0; i<OUTPUT_SIZE; i++) {
		for(int j=0; j<INPUT_SIZE; j++) {
			printf("%Lf ", analytic_grad_X[i][j]);
		}
		printf("\n");
	}
	grad_X = analytic_grad_X;*/

	
	get_output(target_X);
	
	//initialize identity matrix
	for(int i=0; i<INPUT_SIZE; i++) {
		temp_matrix.push_back(vector<LD>());
		for(int j=0; j<INPUT_SIZE; j++) {
			if(i==j) temp_matrix[i].push_back(1);
			else temp_matrix[i].push_back(0);
		}
	}

	const int LAYER_NUM = (int)layer_list.size();
	for(int layer = 0; layer<LAYER_NUM; layer++) {
		grad_X = vector<vector<LD> >();
		grad_X.clear();
		const int LAYER_SIZE = (int)layer_list[layer]->size();
		const int PREVIOUS_LAYER_SIZE = (layer==0) ? INPUT_SIZE : (int)layer_list[layer-1]->size();
		
		for(int i=0; i<LAYER_SIZE; i++) {
			grad_X.push_back(vector<LD>());
			for(int j=0; j<INPUT_SIZE; j++) {
				grad_X[i].push_back(0);
				for(int k=0; k<PREVIOUS_LAYER_SIZE; k++) {
					grad_X[i][j] += (*layer_list[layer])[i]->input_weight_list[k]->get_w() * temp_matrix[k][j];
				}
			}
		}

		for(int i=0; i<LAYER_SIZE; i++) {
			LD deriv = deriv_act((*layer_list[layer])[i]->get_linear_output());
			for(int j=0; j<INPUT_SIZE; j++) {
				grad_X[i][j] *= deriv;
			}
		}

		temp_matrix = vector<vector<LD> >();
		temp_matrix.clear();
		temp_matrix = grad_X;

#ifdef PRINT_MODE
		printf("layer: %d\n", layer);
		for(int i=0; i<LAYER_SIZE; i++) {
			for(int j=0; j<INPUT_SIZE; j++) {
				printf("%.4Lf ", temp_matrix[i][j]);
			}
			printf("\n");
		}
		fflush(stdout);
#endif
	}

	grad_X = vector<vector<LD> >();
	grad_X.clear();
	const int LAYER_SIZE = (int)output_layer->size();
	const int PREVIOUS_LAYER_SIZE = (int)layer_list[(int)layer_list.size()-1]->size();
		
	for(int i=0; i<LAYER_SIZE; i++) {
		grad_X.push_back(vector<LD>());
		for(int j=0; j<INPUT_SIZE; j++) {
			grad_X[i].push_back(0);
			for(int k=0; k<PREVIOUS_LAYER_SIZE; k++) {
				grad_X[i][j] += (*output_layer)[i]->input_weight_list[k]->get_w() * temp_matrix[k][j];
			}
		}
	}

	for(int i=0; i<LAYER_SIZE; i++) {
		LD deriv = deriv_act((*output_layer)[i]->get_linear_output());
		for(int j=0; j<INPUT_SIZE; j++) {
			grad_X[i][j] *= deriv;
		}
	}

#ifdef PRINT_MODE
	printf("layer: output\n");
	for(int i=0; i<LAYER_SIZE; i++) {
		printf("%d: ", i);
		for(int j=0; j<INPUT_SIZE; j++) {
			printf("%.4Lf ", grad_X[i][j]);
		}
		printf("\n");
	}
	fflush(stdout);
#endif
	
}
	
//GD on X
void GDX::update_grad_X(LD learning_rate)
{
	const int INPUT_SIZE = (int)input_layer->size();
	for(int i=0; i<INPUT_SIZE; i++) {
		target_X[i] = target_X[i] + learning_rate * grad_X[TARGET_CLASS][i];
		if(target_X[i] < MIN)
			target_X[i] = MIN;
		else if(target_X[i] > MAX)
			target_X[i] = MAX;
	}
}

//get current X
vector<LD> GDX::get_X()
{
	return target_X;
}

void GDX::set_X(LD range_min, LD range_max, vector<LD> data, int target_class)
{
	TARGET_CLASS = target_class;
	MIN = range_min;
	MAX = range_max;
	target_X = data;
}