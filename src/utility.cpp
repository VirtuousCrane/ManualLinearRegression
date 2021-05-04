#include "utility.hpp"
#include <iostream>
#include <vector>
#include <fstream>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

using namespace Eigen;
using namespace std;

namespace utility{

	InvalidFile::InvalidFile(){
		errMessage = "Invalid File";
	}

	string InvalidFile::what(){
		return errMessage;
	}

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
		vector<double> values;
		uint rows = 0;
		while(getline(indata, line)){
			stringstream lineStream(line);
			string cell;
			while(getline(lineStream, cell, ',')){
				values.push_back( stod(cell));
			}
			++rows;
		}

		return Map<utility::MatrixXfR>(values.data(), rows, values.size()/rows);
	}

	MatrixXfR load_label(const std::string& path){
		ifstream indata;
		indata.open(path);
		string line;
		vector<double> values;
		uint rows = 0;
		while(getline(indata, line)){
			values.push_back( stod(line));
			++rows;
		}

		return Map<utility::MatrixXfR>(values.data(), rows, 1);
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

	int file_exists(string path){
		ifstream f(path.c_str());
		if(f.good()){
			return 1;
		} else {
			return 0;
		}
	}

/*	void write_to_file(string path, MatrixXfR& data){
		ofstream f;
		f.open(path.c_str());

		for(int i = 0; i < data.rows(); i++){
			for(int j = 0; j < data.cols(); j++){
				f << data(i, j) << ",";
			}
			f << endl;
		}

		f.close();
	}
*/

	void write_to_file(string path, MatrixXfR& data){
		const static IOFormat CSVFormat(StreamPrecision, DontAlignCols, ", ", "\n");
		ofstream f(path.c_str());

		if(f.is_open()){
			f << data.format(CSVFormat);
			f.close();
		}
	}

	void write_to_file(string path, double data){
		ofstream f;
		f.open(path.c_str());

		f << data << endl;

		f.close();
	}

	MatrixXfR load_saved_matrix(string path){
		vector<double> matrixEntries;
		ifstream matrixDataFile(path.c_str());

		string matrixRowString;
		string matrixEntry;
		int matrixRowNumber = 0;

		while(getline(matrixDataFile, matrixRowString)){
			stringstream matrixRowStringStream(matrixRowString);
			while(getline(matrixRowStringStream, matrixEntry, ',')){
				matrixEntries.push_back(stod(matrixEntry));
			}
			matrixRowNumber++;
		}
		return Map<utility::MatrixXfR>(matrixEntries.data(), matrixRowNumber, matrixEntries.size()/matrixRowNumber);
	}
}

