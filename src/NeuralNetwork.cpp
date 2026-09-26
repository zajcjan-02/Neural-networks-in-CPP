//
// Created by firep on 23/06/2026.
//

#include "NeuralNetwork.h"

#include <algorithm>
#include <iostream>
#include <ostream>


STATUS NeuralNetwork::add_layer(DenseLayer layer) {
	if (is_compiled) {
		std::cerr << "Cannot add new layer after the neural network was compiled" << std::endl;
		return STATUS::ALREADY_COMPILED;
	}
	layers.push_back(std::move(layer));
	return STATUS::OK;
}


STATUS NeuralNetwork::compile() {
	if (learning_rate > 1 || learning_rate < 0) {
		std::cerr << "ValueError: learning rate must be on the interval [0, 1]" << std::endl;
		return STATUS::VALUE_OUT_OF_RANGE_ERROR;
	}

	if (layers.empty()) {
		std::cerr << "NeuralNetwork::compile(): layers is empty" << std::endl;
		return STATUS::MISCONFIGURATION;
	}

	for (int i = 1; i<layers.size(); i++) {
		layers[i].previous = &layers[i-1];
	}

	is_compiled = true;
	return STATUS::OK;
}


STATUS NeuralNetwork::forward_propagation(const std::vector<float>& input, std::vector<float>& outarr) {
	if (!is_compiled) {
		std::cerr << "Neural network must be compiled first" << std::endl;
		return STATUS::NOT_COMPILED;
	}

	if (layers.empty()) {
		std::cerr << "Network has no layers" << std::endl;
		return STATUS::MISCONFIGURATION;
	}

	// First DenseLayer consumes the raw input vector
	if (!layers[0].forward(input)) {
		return STATUS::PROPAGATION_FAILED;
	}

	// Every next DenseLayer consumes previous layer's output activations
	for (size_t i = 1; i < layers.size(); i++) {
		if (!layers[i].forward(layers[i - 1].activation_values)) {
			return STATUS::PROPAGATION_FAILED;
		}
	}

	outarr = layers.back().activation_values;

	return STATUS::OK;
}

float NeuralNetwork::Loss_(float x, float y) const {
	switch (lossFunction) {
		case LOSS_FUNCTION::MSE:
			return mse_(x, y);

		case LOSS_FUNCTION::CROSS_ENTROPY:
			return y == 0.0f ? 0.0f : -y * std::log(x);

		case LOSS_FUNCTION::RMSE:
		default:
			return x-y;

	}
}



float NeuralNetwork::derivative(float value, const ACTIVATION_FUNCTION& activationFunction) {
	switch (activationFunction) {
		case ACTIVATION_FUNCTION::SIGMOID:
			return sig_derivative(value);
		case ACTIVATION_FUNCTION::RELU:
			return relu_derivative(value);
		case ACTIVATION_FUNCTION::TANH:
			return tanh_derivative(value);
		default:
			std::cerr << "Unknown activation function"  << std::endl;
			return 0;
	}
}

STATUS NeuralNetwork::update_weights(DenseLayer& current, const std::vector<float>& X) const {

	const bool c = current.previous == nullptr;

	size_t pc = !c ? current.previous->activation_values.size(): X.size(); // previous (neuron) count
	size_t cc = current.grad_neurons.size();

	for (int g=0; g < cc; g++) {
		for (int n= 0; n < pc; n++) {
			float av = !c ? current.previous->activation_values[n] : X[n];
			float delta = -learning_rate * current.grad_neurons[g] * av;
			if (n + g * pc >= current.weights.size()) {
				std::cerr << "Weight index out of bounds" << std::endl;
				return STATUS::SIZE_MISMATCH;
			}
			current.weights[g][n] += delta;
		}
	}


	return STATUS::OK;
}


// TODO different loss functions (only mse and categorical cross entropy right now)
STATUS NeuralNetwork::back_propagation(const std::vector<float>& X, const std::vector<float>& y) {

	if (!is_compiled) {
		std::cerr << "Neural network must be compiled first" << std::endl;
		return STATUS::NOT_COMPILED;
	}

	if (layers.back().activation_values.size() != y.size()) {
		std::cerr << "Output layer and expected values array size do not match!" << std::endl;
		return STATUS::SIZE_MISMATCH;
	}
	for (int lidx = static_cast<int>(layers.size() -1); lidx >= 0; lidx--){
		// if output layer
		if (lidx == static_cast<int>(layers.size()-1)) {
			int cc = static_cast<int>(layers[lidx].activation_values.size()); // current (neuron) count
			for (int neuron = 0; neuron < cc; neuron++) {
				float grad;
				float out = layers[lidx].activation_values[neuron];
				if (lossFunction == LOSS_FUNCTION::CROSS_ENTROPY) {
					grad = out - y[neuron];
				}else {
					float err = mse_derivative(out, y[neuron]);
					float der = derivative(out, layers[lidx].activationFunction);
					grad = err * der;
				}
				layers[lidx].grad_neurons[neuron] = grad;
			}
		}
		else { // hidden layers
			size_t nc = layers[lidx + 1].activation_values.size(); // next layer neuron count
			size_t cc = layers[lidx].activation_values.size();     // current layer neuron count
			// @Legacy
			/*for (size_t neuron = 0; neuron < cc; neuron++) {
				float sum = 0.0f;

				for (size_t next = 0; next < nc; next++) {
					size_t widx = next * cc + neuron;

					sum += layers[lidx + 1].grad_neurons[next]
						 * layers[lidx + 1].weights[widx];
				}

				float a_der = derivative(
					layers[lidx].activation_values[neuron],
					layers[lidx].activationFunction
				);

				float grad = sum * a_der;
				layers[lidx].grad_neurons[neuron] = grad;
			}*/

			for (size_t neuron = 0; neuron < cc; neuron++) {
				float sum = 0.0f;

				for (size_t next = 0; next < nc; next++) {
					size_t widx = next * cc + neuron;

					sum += layers[lidx + 1].grad_neurons[next] * layers[lidx + 1].weights[next][neuron];
				}

				float a_der = derivative(
					layers[lidx].activation_values[neuron],
					layers[lidx].activationFunction
				);

				float grad = sum * a_der;
				layers[lidx].grad_neurons[neuron] = grad;
			}
		}

	}

	for (int lidx = static_cast<int>(layers.size() -1); lidx >=0; lidx--) {
		if (update_weights(layers[lidx], X)!= STATUS::OK) {
			std::cerr << "Update weights failed" << std::endl;
			return STATUS::WEIGHT_UPDATE_FAILURE;
		}
	}


	return STATUS::OK;

}


STATUS NeuralNetwork::train(const std::vector<float>& X, const std::vector<float>& y, std::vector<float>& out) {
	std::vector<float> output;
	if (forward_propagation(X, output) != STATUS::OK) {
		return STATUS::PROPAGATION_FAILED;
	}

	out = layers.back().activation_values;

	if (y.size() != output.size()) {
		std::cerr << "Size mismatch output and expected vectors do not match" << std::endl;
		return STATUS::SIZE_MISMATCH;
	}

	if (back_propagation(X, y) != STATUS::OK) {
		std::cerr << "Backpropagation failed" << std::endl;
		return STATUS::BACKPROPAGATE_FAILED;
	}

	if (adam_step() != STATUS::OK) {
		std::cerr << "Adam step failed" << std::endl;
		return STATUS::ADAM_STEP_FAILURE;
	}


	return STATUS::OK;
}

STATUS NeuralNetwork::adam_update(
	Matrix<float> &params,
	const Matrix<float> &grads,
	Matrix<float> &m,
	Matrix<float> &v) const {

	auto beta1_t = static_cast<float>(std::pow(adam.beta1, adam.t));
	auto beta2_t = static_cast<float>(std::pow(adam.beta2, adam.t));

	if (m.size() != params.size()) {
		std::cerr << "Size mismatch: first moment vector m and params vector sizes do not match" << std::endl;
		return STATUS::SIZE_MISMATCH;
	}
	if (v.size() != params.size()) {
		std::cerr << "Size mismatch: second moment vector v and params vector sizes do not match" << std::endl;
		return STATUS::SIZE_MISMATCH;
	}

	for (size_t i = 0; i<params.rows(); i++) {
		for (size_t j = 0; j<params.cols(); j++) {
			float g = grads[i][j];
			m[i][j] = adam.beta1 * m[i][j] * (1.0f - adam.beta1) * g;
			v[i][j] = adam.beta2 * v[i][j] * (1.0f - adam.beta2) * g * g;

			if (1.0f - beta1_t == 0.f) {
				std::cerr << "m_hat calculation caused division by zero";
				return STATUS::DIVISION_BY_ZERO;
			}

			if (1.0f - beta2_t == 0.f) {
				std::cerr << "v_hat calculation caused division by zero";
				return STATUS::DIVISION_BY_ZERO;
			}

			float m_hat = m[i][j] / (1.0f - beta1_t);
			float v_hat = v[i][j] / (1.0f - beta2_t);
			params[i][j] = adam.lr * m_hat / (std::sqrt(v_hat) + adam.eps);
		}
	}

	return STATUS::OK;
}

STATUS NeuralNetwork::adam_update(
	std::vector<float>& params,
	const std::vector<float>& grads,
	std::vector<float>& m,
	std::vector<float>& v
	) const {

	auto beta1_t = static_cast<float>(std::pow(adam.beta1, adam.t));
	auto beta2_t = static_cast<float>(std::pow(adam.beta2, adam.t));

	if (m.size() != params.size()) {
		std::cerr << "Size mismatch: first moment vector m and params vector sizes do not match" << std::endl;
		return STATUS::SIZE_MISMATCH;
	}
	if (v.size() != params.size()) {
		std::cerr << "Size mismatch: second moment vector v and params vector sizes do not match" << std::endl;
		return STATUS::SIZE_MISMATCH;
	}
	for (size_t i = 0; i < params.size(); i++) {
		float g = grads[i];
		m[i] = adam.beta1 * m[i] * (1.0f - adam.beta1) * g;
		v[i] = adam.beta2 * v[i] * (1.0f - adam.beta2) * g * g;

		if (1.0f - beta1_t == 0.f) {
			std::cerr << "m_hat calculation caused division by zero";
			return STATUS::DIVISION_BY_ZERO;
		}

		if (1.0f - beta2_t == 0.f) {
			std::cerr << "v_hat calculation caused division by zero";
			return STATUS::DIVISION_BY_ZERO;
		}

		float m_hat = m[i] / (1.0f - beta1_t);
		float v_hat = v[i] / (1.0f - beta2_t);

		params[i] = adam.lr * m_hat / (std::sqrt(v_hat) + adam.eps);
	}

	return STATUS::OK;
}


STATUS NeuralNetwork::adam_step() {
	adam.t++;

	for (DenseLayer &layer : layers) {
		if (adam_update(
			layer.weights,
			layer.grad_weights,
			layer.m_weights,
			layer.v_weights
			) != STATUS::OK) {
			return STATUS::WEIGHT_UPDATE_FAILURE;
		};
		if (adam_update(
				layer.bias,
				layer.grad_bias,
				layer.m_bias,
				layer.v_bias
				) != STATUS::OK) {
			return STATUS::BIAS_UPDATE_FAILURE;
		};
	}
	return STATUS::OK;
}



STATUS NeuralNetwork::fit(const std::vector<std::vector<float>>& X, const std::vector<std::vector<float>>& y, std::vector<std::vector<float>>& outputs) {

	if (X.size() != y.size()) {
		std::cerr << "Size mismatch: output and expected vectors do not match" << std::endl;
		return STATUS::SIZE_MISMATCH;
	}

	for (auto& x : X) {
		for (auto& value : x) {
			if (std::abs(value) > 1) {
				std::cerr << "Values must lie on the interval [-1, 1]. Received " << value << std::endl;
				return STATUS::VALUE_OUT_OF_RANGE_ERROR;
			}
		}
	}


	for (int x = 0; x < X.size(); x++) {
		if (train(X[x], y[x], outputs[x]) != STATUS::OK) {
			std::cerr << "Training failed" << std::endl;
			return STATUS::TRAINING_FAILURE;
		}
	}

	return STATUS::OK;
}

