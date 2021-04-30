#include "utility.hpp"
#include <iostream>
#include <vector>
#include <fstream>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

using namespace Eigen;
using namespace std;

namespace utility{
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

	void add_constant_to_matrix(utility::MatrixXfR& mat, const double n){
		for(int i=0; i<mat.rows(); i++){
			for(int j=0; j<mat.cols(); j++){
				mat(i, j) = mat(i, j) + n;
			}
		}
	}

	MatrixXfR load_csv(const std::string& path){
		ifstream indata;
		indata.open(path);
		string line;
		vector<float> values;
		uint rows = 0;
		while(getline(indata, line)){
			stringstream lineStream(line);
			string cell;
			while(getline(lineStream, cell, ',')){
				values.push_back((float) stod(cell));
			}
			++rows;
		}
		return Map<MatrixXfR>(values.data(), rows, values.size()/rows);
	}

	MatrixXfR load_label(const std::string& path){
		ifstream indata;
		indata.open(path);
		string line;
		vector<float> values;
		uint rows = 0;
		while(getline(indata, line)){
			values.push_back((float) stod(line));
			++rows;
		}
		return Map<MatrixXfR>(values.data(), rows, 1);
	}

	template<typename Base, typename T>
	inline bool instanceof(const T*){
		return is_base_of<Base, T>::value;
	}

	utility::MatrixXfR init_weights(int X_size, int hidden_layer_size){
		utility::MatrixXfR output;
		output = utility::MatrixXfR::Random(X_size, hidden_layer_size);
		return output;
	}

	double init_bias(){
		return rand() % (10 + 1);
	}
}

