#ifndef __GAN__
#define __GAN__

/*#include <vector>
#include <functional>
#include "node.h"*/
#include "generator.h"
#include "discriminator.h"

#define LD long double

typedef vector<Node*> Layer;
typedef vector<long double> Data;

//class Discriminator;
//class Generator;

class GAN
{
private:
	Discriminator D;
	Generator G;
	function<LD(LD)> deriv_act_D;
	void initialize_discriminator();
	bool D_initialized;
public:
	GAN();

	void add_generator_layer(int num, function<LD(LD)> act, function<LD(LD)> deriv_act);
	void add_generator_input_layer(int num);
	void add_generator_output_layer(int num);
	void add_generator_all_weights();

	void add_discriminator_layer(int num, function<LD(LD)> act, function<LD(LD)> deriv_act);
	void add_discriminator_all_weights();

	Data get_generator_output();
	Data get_generator_output(Data& input_data);
	pair<LD, Data> get_discriminator_output_from_random(Data& random_data);
	LD get_discriminator_output(Data& input_data);

	void train_generator(LD learning_rate);
	void train_generator(LD learning_rate, int num);
	void train_discriminator(LD learning_rate, Data& input_data, Data& output_data);
	void train_discriminator(LD learning_rate, vector<Data>& input_data, vector<Data>& output_data);
	void train(LD learning_rate, int d_num, int g_num, vector<Data>& input_data);

	void print_bias_and_weights();
};

#endif