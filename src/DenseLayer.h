#pragma once

//
// Created by firep on 16/06/2026.
//

#define NEURALNETWORKSC_DENSELAYER_H
#include <cstddef>
#include <vector>
#include <cmath>
enum class ACTIVATION_FUNCTION {
	RELU = 0,
	SIGMOID = 1,
	TANH = 2,
	SOFTMAX = 3,
};



class DenseLayer {
	public:
		std::size_t in;
		std::size_t out;
		DenseLayer(size_t n_in, size_t n_out);
		ACTIVATION_FUNCTION activationFunction = ACTIVATION_FUNCTION::RELU;
		DenseLayer* previous = nullptr;
		std::vector<float> weights;      // out x in
		std::vector<float> bias;         // out
		std::vector<float> grad_neurons; // out
		std::vector<float> grad_weights; // out x in
		std::vector<float> grad_bias;    // out
		std::vector<float> activation_values;
		bool forward(const std::vector<float>& previous_activations);
		std::vector<float> delta;
		std::vector<float> m_weights;
		std::vector<float> m_bias;

		std::vector<float> v_weights;
		std::vector<float> v_bias;
		std::vector<float> output;




	private:
		void xavier_weight_initialization();
		float activation(float value);

		static float sig_(float value) { return 1.f / (1 + std::exp(-value)); }
		static float tanh_(float value) { return (std::exp(value) - std::exp(-value)) / (std::exp(value) + std::exp(-value)); }
		static float relu_(float value) { return value < 0 ? 0 : value; }
		int timestamp=0;



};
