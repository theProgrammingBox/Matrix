#include "Matrix.h"

int main() 
{
	int inputs = 2;
	int outputs = 3;
	int batchSize = 100;
	float invBatchSize = 1.0f / batchSize;
	float learningRate = 0.1f;
	
	Matrix<float> input(1, inputs);
	Matrix<float> weights(inputs, outputs);
	Matrix<float> bias(1, outputs);
	Matrix<float> output(1, outputs);
	Matrix<float> outputActivated(1, outputs);
	
	auto nothing = [](float x) { return x; };
	auto nothingGradient = [](float x, float y) { return y; };
	auto leakyRelu = [](float x) { return x * (1.0f - (x < 0.0f) * 0.9f); };
	auto leakyReluGradient = [](float x, float y) { return y * (1.0f - (x < 0.0f) * 0.9f); };
	auto relu = [](float x) { return x * (x > 0.0f); };
	auto reluGradient = [](float x, float y) { return y * (x > 0.0f); };

	auto activation = relu;
	auto activationGradient = reluGradient;
	
	weights.fillRandom();
	bias.fillRandom();

	Matrix<float> expected(1, outputs);
	Matrix<float> outputActivatedGradient(1, outputs);
	Matrix<float> outputGradient(1, outputs);
	Matrix<float> weightsGradient(inputs, outputs);
	
	int iter = 200;
	while (iter--)
	{
		outputGradient.fill(0.0f);
		weightsGradient.fill(0.0f);
		for (int i = 0; i < batchSize; i++)
		{
			input.fillRandom();
			output = input * weights + bias;
			outputActivated = output;
			outputActivated.apply(activation);

			for (int i = 0; i < outputs; i++)
			{
				expected(0, i) = input(0, 0) * (i * 0.2 - 0.3) - input(0, 1) * (i * 1.4 - 1.6) + i - 0.3;
			}
			expected.apply(activation);

			/*
			-0.3, -0.1, 0.1
			1.6, 0.2, -1.2

			-0.3, 0.7, 1.7
			*/

			outputActivatedGradient = expected - outputActivated;
			outputGradient.addOn(activationGradient, output, outputActivatedGradient);

			input.transpose();
			weightsGradient += input * outputGradient;
			input.transpose();

			/*cout << "input:\n";
			input.print();
			cout << "weights:\n";
			weights.print();
			cout << "bias:\n";
			bias.print();
			cout << "output:\n";
			output.print();*/
			/*cout << "outputActivated:\n";
			outputActivated.print();
			cout << "expected:\n";
			expected.print();*/
			/*cout << "outputActivatedGradient:\n";
			outputActivatedGradient.print();*/
		}
		weightsGradient *= invBatchSize * learningRate;
		weights += weightsGradient;
		outputGradient *= invBatchSize * learningRate;
		bias += outputGradient;

		/*cout << "outputGradient:\n";
		outputGradient.print();
		cout << "weightsGradient:\n";
		weightsGradient.print();*/
	}

	cout << "weights:\n";
	weights.print();
	cout << "bias:\n";
	bias.print();

	return 0;
}