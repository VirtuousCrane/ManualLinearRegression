#include <iostream>
#include <vector>
#include <fstream>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#ifndef UTILITY_HPP
#define UTILITY_HPP

using namespace Eigen;
using namespace std;

namespace utility{
	typedef Matrix<double, Dynamic, Dynamic, RowMajor> MatrixXfR;

	void element_wise_exp(MatrixXfR& mat);
	void element_wise_inverted_division(MatrixXfR& mat, float divisor);
	void element_wise_power(MatrixXfR& mat, int p);
	void add_vector_to_matrix(MatrixXfR& vec, MatrixXfR& mat);
	void add_constant_to_matrix(MatrixXfR& mat, const double n);

	MatrixXfR load_csv(const string& path);
	MatrixXfR load_label(const string& path);

	MatrixXfR init_weights(int X_size, int hidden_layer_size);
	double init_bias();

	void apply_cyclic_learning_rate(double min, double max, double& current, double incrementor);

	template<typename Base, typename T>
	inline bool instanceof(const T*);
}

#endif
