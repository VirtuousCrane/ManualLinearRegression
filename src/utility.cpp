#include "utility.hpp"
#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

using namespace Eigen;
using namespace std;

void element_wise_exp(utility::MatrixXfR& mat){
	for(int i=0; i<mat.rows(); i++){
		for(int j=0; j<mat.cols(); j++){
			mat(i, j) = exp(mat(i, j));
		}
	}
}

void element_wise_inverted_division(utility::MatrixXfR& mat, float divisor){
	for(int i=0; i<mat.rows(); i++){
		for(int j=0; j<mat.cols(); j++){
			mat(i, j) = divisor/mat(i, j);
		}
	}
}

void element_wise_power(utility::MatrixXfR& mat, int p){
	for(int i=0; i<mat.rows(); i++){
		for(int j=0; j<mat.cols(); j++){
			mat(i, j) = pow(mat(i, j), p);
		}
	}
}


void add_vector_to_matrix(utility::MatrixXfR& vec, utility::MatrixXfR& mat){
	for(int i=0; i<mat.rows(); i++){
		for(int j=0; j<mat.cols(); j++){
			mat(i, j) = mat(i, j) + vec(0, j);
		}
	}
}

template<typename Base, typename T>
inline bool instanceof(const T*){
	return is_base_of<Base, T>::value;
}
