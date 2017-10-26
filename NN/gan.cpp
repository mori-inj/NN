#include "gan.h"
//#include "generator.h"
//#include "discriminator.h"

GAN::GAN()
{
	D_initialized = false;
}

void GAN::add_generator_layer(int num, function<LD(LD)> act, function<LD(LD)> deriv_act)
{
	G.add_layer(num, act, deriv_act);
}

void GAN::add_generator_input_layer(int num)
{
	G.add_input_layer(num);
}

void GAN::add_generator_output_layer(int num)
{
	G.add_output_layer(num);
}

void GAN::add_generator_all_weights()
{
	G.add_all_weights();
	G.dump_weights();
}


void GAN::initialize_discriminator()
{
	D.add_input_layer(G.get_output_size());
	D.add_output_layer(1);
}

void GAN::add_discriminator_layer(int num, function<LD(LD)> act, function<LD(LD)> deriv_act)
{
	if(!D_initialized) {
		initialize_discriminator();
		D_initialized = true;
	}
	deriv_act_D = deriv_act;
	D.add_layer(num, act, deriv_act);
}

void GAN::add_discriminator_all_weights()
{
	D.add_all_weights();
}


Data GAN::get_generator_output()
{
	Data random_data;
	for(int i=0; i<(int)G.get_input_size(); i++) {
		random_data.push_back(rand()%256);
	}
	return G.get_output(random_data);
}

Data GAN::get_generator_output(Data& input_data)
{
	return G.get_output(input_data);
}

pair<LD, Data> GAN::get_discriminator_output_from_random(Data& random_data)
{
	Data generated = G.get_output(random_data);
	LD ret = D.get_output(generated)[0];

	return make_pair(ret, generated);
}

LD GAN::get_discriminator_output(Data& input_data)
{
	return D.get_output(input_data)[0];
}


void GAN::train_generator(LD learning_rate)
{
	Data random_data;
	for(int i=0; i<(int)G.get_input_size(); i++) {
		random_data.push_back(rand()%256);
	}
	G.train(
		learning_rate, 
		random_data, 
		D.calc_grad_X(random_data, deriv_act_D), 
		D.get_output(G.get_output(random_data))[0]
	); 
}

void GAN::train_generator(LD learning_rate, int num)
{
	for(int i=0; i<num; i++) {
		train_generator(learning_rate);
	}
}

void GAN::train_discriminator(LD learning_rate, Data& input_data, Data& output_data)
{
	D.train(learning_rate, input_data, output_data);
}

void GAN::train_discriminator(LD learning_rate, vector<Data>& input_data, vector<Data>& output_data)
{
	D.train(learning_rate, input_data, output_data);
}

void GAN::train(LD learning_rate, int d_num, int g_num, vector<Data>& input_data)
{
	const int INPUT_SIZE = (int)input_data.size();
	for(int cnt=0; cnt<INPUT_SIZE;) {
		for(int i=0; i<d_num && cnt<INPUT_SIZE; i++) {
			if(rand()%2) {
				train_discriminator(learning_rate, input_data[cnt++], Data(1, 1));
			} else {
				train_discriminator(learning_rate, get_generator_output(), Data(1, 0));
			}
		}
		train_generator(learning_rate, g_num);
	}
}


void GAN::print_bias_and_weights()
{
}