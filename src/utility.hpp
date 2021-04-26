#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#ifndef UTILITY_HPP
#define UTILITY_HPP

using namespace Eigen;
using namespace std;

namespace utility{
	typedef Matrix<float, Dynamic, Dynamic, RowMajor> MatrixXfR;

	void element_wise_exp(MatrixXfR& mat);
	void element_wise_inverted_division(MatrixXfR& mat, float divisor);
	void element_wise_power(MatrixXfR& mat, int p);
	void add_vector_to_matrix(MatrixXfR& vec, MatrixXfR& mat);

	template<typename Base, typename T>
	inline bool instanceof(const T*);
}

#endif
