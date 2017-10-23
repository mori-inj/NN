#include "discriminator.h"
#include "weight.h"

//#define PRINT_MODE

//get gradient on X
vector<Data> Discriminator::calc_grad_X(Data input_data, function<LD(LD)> deriv_act)
{
	vector<vector<LD> > temp_matrix;
	const int INPUT_SIZE = (int)input_layer->size();
	const int OUTPUT_SIZE = (int)output_layer->size();
	
	get_output(input_data);
	
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
	return grad_X;	
}