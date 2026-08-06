//
// Created by firep on 16/06/2026.
//

#include "DenseLayer.h"

#include <cmath>
#include <iostream>
#include <random>


DenseLayer::DenseLayer(size_t n_in, size_t n_out): in(n_in), out(n_out) {
	weights.resize(n_out * n_in);
	bias.resize(n_out);
	grad_weights.resize(n_out * n_in);
	grad_bias.resize(n_out);
	grad_neurons.resize(n_out);
	activation_values.resize(n_out);
	xavier_weight_initialization();

	// Adam
	m_weights.assign(weights.size(), 0.0f);
	m_bias.assign(bias.size(), 0.0f);
	v_weights.assign(weights.size(), 0.0f);
	v_bias.assign(bias.size(), 0.0f);


}


float DenseLayer::activation(float value) {
	// TODO add other activation functions as well
	switch (activationFunction) {
		case ACTIVATION_FUNCTION::SIGMOID:
			return sig_(value);
		case ACTIVATION_FUNCTION::TANH:
			return tanh_(value);
		case ACTIVATION_FUNCTION::RELU:
			return relu_(value);
		case ACTIVATION_FUNCTION::SOFTMAX:
			return value;
		default:
			return relu_(value);
	}
}




bool DenseLayer::forward(const std::vector<float>& previous_activations) {
	if (previous_activations.size() != in) {
		std::cerr << "Input size mismatch. Expected "
				  << in << ", got " << previous_activations.size() << std::endl;
		return false;
	}

	if (weights.size() != in * out) {
		std::cerr << "Weight size mismatch. Expected "
				  << in * out << ", got " << weights.size() << std::endl;
		return false;
	}

	if (bias.size() != out) {
		std::cerr << "Bias size mismatch\n";
		return false;
	}

	activation_values.resize(out);

	for (size_t neuron = 0; neuron < out; neuron++) {
		float val = bias[neuron];

		for (size_t input = 0; input < in; input++) {
			size_t w = neuron * in + input;
			val += previous_activations[input] * weights[w];
		}
		activation_values[neuron] = activation(val);


	}
	if (activationFunction == ACTIVATION_FUNCTION::SOFTMAX) {
		float s = 0;
		for (int i = 0; i < activation_values.size(); i++) {
			s += std::exp(activation_values[i]);
		}

		for (auto& val : activation_values) {
			val = std::exp(val) / s;
		}
	}

	return true;
}

void DenseLayer::xavier_weight_initialization() {
	float limit = std::sqrt(6.0f / static_cast<float>(in + out));

	limit = std::min(limit, 1.0f);

	static std::random_device rd;
	static std::mt19937 gen(rd());

	std::uniform_real_distribution<float> dist(-limit, limit);

	for (float& w : weights) {
		w = dist(gen);
	}

	for (float& b : bias) {
		b = 0.0f;
	}
}

