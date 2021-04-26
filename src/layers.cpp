#include <iostream>
#include "utility.hpp"
#include "layers.hpp"
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;
using namespace utility;
using namespace layers;

// ===============================================================================
//   Define: DifferentDimension Class
// ===============================================================================

DifferentDimension::DifferentDimension(){
	errMessage = "Different Dimension";
}

string DifferentDimension::what(){
	return errMessage;
}

// ===============================================================================
//   Define: [Abstract Class] Operation
// ===============================================================================

Operation::Operation(const MatrixXfR& x){
	X_ = x;
}


// ===============================================================================
//   Define: [Abstract Class] Loss
// ===============================================================================

Loss::Loss(const MatrixXfR& p, const MatrixXfR& t){
	prediction_ = p;
	target_     = t;
}

// ===============================================================================
//   Define: FullyConnected Class
// ===============================================================================

FullyConnected::FullyConnected(const MatrixXfR& x, int hidden_size): Operation(x){
	Weight_ = init_weights(X_.cols(), hidden_size);
	Bias_   = init_bias(hidden_size);
}

void FullyConnected::set_weight(const MatrixXfR& weight){
	Weight_ = weight;
}

void FullyConnected::set_bias(const MatrixXfR& bias){
	Bias_ = bias;
}

string FullyConnected::get_type(){
	return "FullyConnected";
}

MatrixXfR FullyConnected::forward(){
	MatrixXfR temp;

	temp = X_ * Weight_;
	add_vector_to_matrix(Bias_, temp);

	Output_ = temp;

	return Output_;
}

void FullyConnected::backward(){
/*
Let:
  - N = X*W
  - P = N + B
  - dPdN = ones_like(N)
  - dPdB = ones_like(B)
*/
	MatrixXfR dPdN, dPdB;
	MatrixXfR dNdW, dPdW;

	dPdN = MatrixXfR::Ones(X_.cols(), Weight_.rows());
	dPdB = MatrixXfR::Ones(Bias_.rows(), Bias_.cols());

	dNdW = X_.transpose();
	dPdW = dNdW * dPdN;

	dPdB_ = dPdB;
	gradient_ = dPdW;
}

MatrixXfR FullyConnected::get_gradient(){
	return gradient_;
}

MatrixXfR FullyConnected::get_bias_gradient(){
	return dPdB_;
}

// ===============================================================================
//   Define: Sigmoid Class
// ===============================================================================

Sigmoid::Sigmoid(const MatrixXfR& x): Operation(x){};

string Sigmoid::get_type(){
	return "Sigmoid";
}

MatrixXfR Sigmoid::forward(){
	MatrixXfR output;
	output = -1.0 * X_;
	element_wise_exp(output);
	output = MatrixXfR::Ones(output.rows(), output.cols()) + output;
	element_wise_inverted_division(output, 1);
	Output_ = output;
	return Output_;
}

void Sigmoid::backward(){
/*
The derivative of Sigmoid with respect to the input is:
dSdX = S(x) * (1-S(x))
*/
	gradient_ = Output_.cwiseProduct(
				MatrixXfR::Ones(
					Output_.rows(),
					Output_.cols()
					) - Output_
				);
}

MatrixXfR Sigmoid::get_gradient(){
	return gradient_;
}

// ===============================================================================
//   Define: MeanSquaredError Class
// ===============================================================================

MeanSquaredError::MeanSquaredError(const MatrixXfR &p, const MatrixXfR &t): Loss(p, t){};

float MeanSquaredError::forward(){
	float loss;
	MatrixXfR temp;

	temp = prediction_ - target_;
	element_wise_power(temp, 2);

	loss = temp.sum();
	loss /= prediction_.rows();

	loss_ = loss;

	return loss_;
}

void MeanSquaredError::backward(){
	MatrixXfR temp;
	temp = prediction_ - target_;
	temp *= 2.0;
	temp /= prediction_.rows();
	gradient_ = temp;
}

MatrixXfR MeanSquaredError::get_gradient(){
	return gradient_;
}

string get_type(){
	return "MSE";
}

// ===============================================================================
//   Define: Auxiliary functions
// ===============================================================================

MatrixXfR init_weights(int X_size, int hidden_layer_size){
	MatrixXfR output;
	output = MatrixXfR::Random(X_size, hidden_layer_size);
	return output;
}

MatrixXfR init_bias(int hidden_size){
	MatrixXfR output;
	output = MatrixXfR::Random(1, hidden_size);
	return output;
}
