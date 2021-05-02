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
				double temp = exp(mat(i, j));
				temp = min(temp, 0.9999);
				temp = max(temp, 0.0001);
				mat(i, j) = temp;
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

	void apply_relu(const utility::MatrixXfR& data, utility::MatrixXfR& output){
		output = MatrixXfR::Zero(data.rows(), data.cols());
		for(int i = 0; i < data.rows(); i++){
			for(int j = 0; j < data.cols(); j++){
				if(data(i, j) > 0){
					output(i, j) = data(i, j);
				}else{
					output(i, j) = 0;
				}
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

	void normalize_cols(MatrixXfR& mat){
		for(int i = 0; i < mat.cols(); i++){
			mat.col(i).normalize();
		}
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

