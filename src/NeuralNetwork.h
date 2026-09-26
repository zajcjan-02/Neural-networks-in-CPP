#pragma once
//
// Created by firep on 23/06/2026.
//

#include <vector>
#include "STATUS.h"
#include "DenseLayer.h"
enum class LOSS_FUNCTION {
	MSE,
	RMSE,
	CROSS_ENTROPY,
};


class NeuralNetwork {
	public:
		std::vector<DenseLayer> layers;
		LOSS_FUNCTION lossFunction = LOSS_FUNCTION::MSE;
		NeuralNetwork(){};
		STATUS compile();
		STATUS add_layer(DenseLayer layer);
		float learning_rate = .01f;
		STATUS forward_propagation(const std::vector<float> &input, std::vector<float>& output);
		STATUS back_propagation(const std::vector<float>& X, const std::vector<float>& y);
		STATUS fit(const std::vector<std::vector<float>>& X, const std::vector<std::vector<float>>& y, std::vector<std::vector<float>>& outputs);
		float beta1 = 0.9; //
		float beta2 = 0.999;


		struct Adam {
			float beta1 = 0.9f; // The First Moment Estimate
			float beta2 = 0.999f; // The Adaptive Learning Rate
			int t = 0; // Timestamp
			float eps=1e-8f;
			float lr = 0.001f;
		} adam;

	private:
		bool is_compiled = false;
		float Loss_(float x, float y) const;
		static float mse_(float x, float y)  {return static_cast<float>(std::pow(y - x, 2)/2); };
		static float mse_derivative(float x, float y) {return x - y;}
		static float relu_derivative(float value) {return value > 0 ? 1 : 0;};
		static float sig_(float value) { return 1 / (1 + std::exp(-value)); }
		static float tanh_derivative(float value) {return 1  - std::pow(std::tanh(value), 2);};
		static float sig_derivative(float value) {const float s = sig_(value); return s * (1 - s); };
		static float derivative(float value, const ACTIVATION_FUNCTION& activationFunction);
		STATUS update_weights(DenseLayer& current, const std::vector<float>& X) const;
		STATUS train(const std::vector<float>& X, const std::vector<float>& y, std::vector<float>& out);

		STATUS adam_step();
		STATUS adam_update(
			Matrix<float>& params,
			const Matrix<float>& grads,
			Matrix<float>& m,
			Matrix<float>& v
		) const;

		STATUS adam_update(
			std::vector<float>& params,
			const std::vector<float>& grads,
			std::vector<float>& m,
			std::vector<float>& v
		) const;
};


