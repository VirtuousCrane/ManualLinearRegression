// Written By: Chayut Liewlom (63011134),
// the LEGENDARY LEGEND chosen by, uh... , no one

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

	class InvalidFile {
		private:
			string errMessage;
		public:
			InvalidFile();
			string what();
	};

	void element_wise_exp(MatrixXfR& mat);
	void element_wise_inverted_division(MatrixXfR& mat, float divisor);
	void element_wise_power(MatrixXfR& mat, int p);
	void add_vector_to_matrix(MatrixXfR& vec, MatrixXfR& mat);
	void add_constant_to_matrix(MatrixXfR& mat, const double n);
	void apply_relu(const MatrixXfR& data, MatrixXfR& output);
	void normalize_cols(MatrixXfR& mat);

	MatrixXfR load_csv(const string& path);
	MatrixXfR load_saved_matrix(string path);
	MatrixXfR load_label(const string& path);

	MatrixXfR init_weights(int X_size, int hidden_layer_size);
	double init_bias();

	template<typename Base, typename T>
	inline bool instanceof(const T*);

	int file_exists(string path);
	void write_to_file(string path, MatrixXfR& data);
	void write_to_file(string path, double data);
}

#endif
