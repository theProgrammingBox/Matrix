#include "Header.h"

template <typename T>
struct Matrix
{
	T* data;
	uint32_t rows;
	uint32_t cols;
	bool transposed;

	Matrix() : data(nullptr), rows(0), cols(0), transposed(false) {}
	Matrix(uint32_t rows, uint32_t cols) : data(new T[rows * cols]), rows(rows), cols(cols), transposed(false) {}
	Matrix(const Matrix& other) : data(new T[other.rows * other.cols]), rows(other.rows), cols(other.cols), transposed(other.transposed)
	{
		memcpy(data, other.data, rows * cols * sizeof(T));
	}
	Matrix(Matrix&& other) noexcept : data(other.data), rows(other.rows), cols(other.cols), transposed(other.transposed)
	{
		other.data = nullptr;
		other.rows = 0;
		other.cols = 0;
		other.transposed = false;
	}
	~Matrix()
	{
		delete[] data;
	}
	
	Matrix& operator=(const Matrix& other)
	{
		if (this != &other)
		{
			delete[] data;
			data = new T[other.rows * other.cols];
			rows = other.rows;
			cols = other.cols;
			transposed = other.transposed;
			memcpy(data, other.data, rows * cols * sizeof(T));
		}
		return *this;
	}
	Matrix& operator=(Matrix&& other) noexcept
	{
		if (this != &other)
		{
			delete[] data;
			data = other.data;
			rows = other.rows;
			cols = other.cols;
			transposed = other.transposed;
			other.data = nullptr;
			other.rows = 0;
			other.cols = 0;
			other.transposed = false;
		}
		return *this;
	}
	
	T& operator()(uint32_t row, uint32_t col)
	{
		return data[row * (transposed * cols + !transposed) + col * (!transposed * rows + transposed)];
	}
	const T& operator()(uint32_t row, uint32_t col) const
	{
		return data[row * (transposed * cols + !transposed) + col * (!transposed * rows + transposed)];
	}
	
	Matrix operator+(const Matrix& other) const
	{
		assert(rows == other.rows && cols == other.cols);
		Matrix result(rows, cols);
		for (uint32_t i = 0; i < rows * cols; i++)
		{
			result.data[i] = data[i] + other.data[i];
		}
		return result;
	}
	Matrix operator-(const Matrix& other) const
	{
		assert(rows == other.rows && cols == other.cols);
		Matrix result(rows, cols);
		for (uint32_t i = 0; i < rows * cols; i++)
		{
			result.data[i] = data[i] - other.data[i];
		}
		return result;
	}
	Matrix operator*(const Matrix& other) const
	{
		assert(cols == other.rows);
		Matrix result(rows, other.cols);
		for (uint32_t i = 0; i < rows; i++)
		{
			for (uint32_t j = 0; j < other.cols; j++)
			{
				T sum = 0;
				for (uint32_t k = 0; k < cols; k++)
				{
					sum += (*this)(i, k) * other(k, j);
				}
				result(i, j) = sum;
			}
		}
		return result;
	}
	Matrix operator*(T scalar) const
	{
		Matrix result(rows, cols);
		for (uint32_t i = 0; i < rows * cols; i++)
		{
			result.data[i] = data[i] * scalar;
		}
		return result;
	}
	Matrix operator/(T scalar) const
	{
		T invScalar = 1 / scalar;
		Matrix result(rows, cols);
		for (uint32_t i = 0; i < rows * cols; i++)
		{
			result.data[i] = data[i] * invScalar;
		}
		return result;
	}
	
	Matrix& operator+=(const Matrix& other)
	{
		assert(rows == other.rows && cols == other.cols);
		for (uint32_t i = 0; i < rows * cols; i++)
		{
			data[i] += other.data[i];
		}
		return *this;
	}
	Matrix& operator-=(const Matrix& other)
	{
		assert(rows == other.rows && cols == other.cols);
		for (uint32_t i = 0; i < rows * cols; i++)
		{
			data[i] -= other.data[i];
		}
		return *this;
	}
	Matrix& operator*=(const Matrix& other)
	{
		assert(cols == other.rows);
		Matrix result(rows, other.cols);
		for (uint32_t i = 0; i < rows; i++)
		{
			for (uint32_t j = 0; j < other.cols; j++)
			{
				T sum = 0;
				for (uint32_t k = 0; k < cols; k++)
				{
					sum += (*this)(i, k) * other(k, j);
				}
				result(i, j) = sum;
			}
		}
		return *this;
	}
	Matrix& operator*=(T scalar)
	{
		for (uint32_t i = 0; i < rows * cols; i++)
		{
			data[i] *= scalar;
		}
		return *this;
	}
	Matrix& operator/=(T scalar)
	{
		T invScalar = 1 / scalar;
		for (uint32_t i = 0; i < rows * cols; i++)
		{
			data[i] *= invScalar;
		}
		return *this;
	}

	void transpose()
	{
		transposed = !transposed;
		swap(rows, cols);
	}
	
	void print() const
	{
		for (uint32_t i = 0; i < rows; i++)
		{
			for (uint32_t j = 0; j < cols; j++)
			{
				cout << (*this)(i, j) << " ";
			}
			cout << endl;
		}
		cout << endl;
	}

	void fill(T value)
	{
		memset(data, value, rows * cols * sizeof(T));
	}
	
	void fillRandom()
	{
		for (uint32_t i = 0; i < rows * cols; i++)
		{
			data[i] = random.normalRand();
		}
	}
};

int main() 
{
	int inputs = 2;
	int outputs = 3;
	
	Matrix<double> input(1, inputs);
	Matrix<double> output;
	Matrix<double> weights(inputs, outputs);
	Matrix<double> bias(1, outputs);
	
	weights.fillRandom();
	bias.fillRandom();

	Matrix<double> expected(1, outputs);
	Matrix<double> outputGradient;
	Matrix<double> weightsGradient;
	
	while (true)
	{
		input.fillRandom();
		output = input * weights + bias;
		cout << "input:\n";
		input.print();
		cout << "weights:\n";
		weights.print();
		cout << "bias:\n";
		bias.print();
		cout << "output:\n";
		output.print();

		for (int i = 0; i < outputs; i++)
		{
			expected(0, i) = input(0, 0) * (i * 0.2 - 0.3) - input(0, 1) * (i * 1.4 - 1.6) + i - 0.3;
		}
		cout << "expected:\n";
		expected.print();

		outputGradient = expected - output;
		cout << "outputGradient:\n";
		outputGradient.print();
		
		input.transpose();
		weightsGradient = input * outputGradient;
		input.transpose();
		cout << "weightsGradient:\n";
		weightsGradient.print();
		
		weightsGradient *= 0.1;
		weights += weightsGradient;

		outputGradient *= 0.1;
		bias += outputGradient;

		getchar();
	}

	return 0;
}