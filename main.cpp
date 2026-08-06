#include <cstdio>
#include <vector>

#include "src/NeuralNetwork.h"

int main() {
	std::vector<std::vector<float>> input = {
		{0.5f, 0.1f, 0.7f},
		{0.5f, 0.1f, 0.7f},
		{0.5f, 0.1f, 0.7f}

	};
	int r = static_cast<int>(input.size());
	int c = static_cast<int>(input[0].size());

	std::vector<std::vector<float>> outputs(r, std::vector<float>(c, 0));

	std::vector<std::vector<float>> expected = {
		{1.0f},
		{1.0f},
		{1.0f}
	};

	std::vector<float> weights_0 = {
		0.3f, 0.7f, 0.2f,
		0.1f, 0.6f, 0.8f
	};

	std::vector<float> weights_1 = {
		0.2f, 0.1f,
		0.3f, 0.7f
	};

	std::vector<float> weights_2 = {
		0.5f, 0.4f
	};

	NeuralNetwork ann;

	DenseLayer hidden_1(3, 2);
	hidden_1.weights = weights_0;

	DenseLayer hidden_2(2, 2);
	hidden_2.weights = weights_1;

	DenseLayer output(2, 1);
	output.weights = weights_2;
	output.activationFunction = ACTIVATION_FUNCTION::SIGMOID;

	ann.add_layer(hidden_1);
	ann.add_layer(hidden_2);
	ann.add_layer(output);

	ann.compile();

	ann.fit(input, expected, outputs);

	return 0;
}