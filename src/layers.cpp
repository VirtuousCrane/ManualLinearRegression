#include <iostream>
#include "utility.hpp"
#include "layers.hpp"
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;
using namespace utility;

namespace layers{
	// ===============================================================================
	//   Define: DifferentDimension Class
	// ===============================================================================

	DifferentDimension::DifferentDimension(){
		errMessage = "Different Dimension";
	}

	string DifferentDimension::what(){
		return errMessage;
	}

	DifferentDimension::~DifferentDimension(){}

	// ===============================================================================
	//   Define: [Abstract Class] Operation
	// ===============================================================================

	Operation::Operation(){};
	Operation::Operation(const MatrixXfR& x){
		X_ = x;
	}
	Operation::~Operation(){}

	// ===============================================================================
	//   Define: [Abstract Class] Loss
	// ===============================================================================

	Loss::Loss(){}
	Loss::~Loss(){}
	Loss::Loss(const MatrixXfR& p, const MatrixXfR& t){
		prediction_ = p;
		target_     = t;
	}

	// ===============================================================================
	//   Define: FullyConnected Class
	// ===============================================================================

	FullyConnected::FullyConnected(){};
	FullyConnected::~FullyConnected(){};

	FullyConnected::FullyConnected(const MatrixXfR& x, int hidden_size): Operation(x){
		Weight_ = init_weights(X_.cols(), hidden_size);
		Bias_   = init_bias();
	}

	FullyConnected::FullyConnected(const FullyConnected& other){
		X_        = other.X_       ;
		Bias_     = other.Bias_    ;
		dPdB_     = other.dPdB_    ;
		Weight_   = other.Weight_  ;
		Output_   = other.Output_  ;
		gradient_ = other.gradient_;
	}

	FullyConnected& FullyConnected::operator=(const FullyConnected& other){
		X_        = other.X_       ;
		Bias_     = other.Bias_    ;
		dPdB_     = other.dPdB_    ;
		Weight_   = other.Weight_  ;
		Output_   = other.Output_  ;
		gradient_ = other.gradient_;

		return *this;
	}

	void FullyConnected::set_weight(const MatrixXfR& weight){
		Weight_ = weight;
	}

	void FullyConnected::set_bias(const double bias){
		Bias_ = bias;
	}

	MatrixXfR FullyConnected::get_weight(){
		return Weight_;
	}

	double FullyConnected::get_bias(){
		return Bias_;
	}

	string FullyConnected::get_type(){
		return "FullyConnected";
	}

	MatrixXfR FullyConnected::forward(){
		MatrixXfR temp;

		temp = X_ * Weight_;
		add_constant_to_matrix(temp, Bias_);

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
		MatrixXfR dNdW, dNdX, dPdW, dPdX;

		dPdN = MatrixXfR::Ones(X_.rows(), Weight_.cols());
		dPdB = MatrixXfR::Ones(1, 1);

		dNdW = X_.transpose();
		dNdX = Weight_.transpose();

		dPdW = dNdW;
		dPdX = dNdX;

		dPdB_ = dPdB;
		dPdX_ = dPdX;
		gradient_ = dPdW;
	}

	void FullyConnected::update(const MatrixXfR& gradW, const MatrixXfR& gradB, double learning_rate){
		Weight_ -= learning_rate * gradW;
		Bias_   -= learning_rate * gradB(0, 0);
	}

	MatrixXfR FullyConnected::get_gradient(){
		return gradient_;
	}

	MatrixXfR FullyConnected::get_x_gradient(){
		return dPdX_;
	}

	MatrixXfR FullyConnected::get_bias_gradient(){
		return dPdB_;
	}

	void FullyConnected::get_weight_dimension(){
		cout << "Weight dim: " << Weight_.rows() << " " << Weight_.cols() << endl;
	}

	// ===============================================================================
	//   Define: Sigmoid Class
	// ===============================================================================

	Sigmoid::Sigmoid(){};
	Sigmoid::~Sigmoid(){};

	Sigmoid::Sigmoid(const MatrixXfR& x): Operation(x){}
	Sigmoid::Sigmoid(const Sigmoid& other){
		X_        = other.X_       ;
		gradient_ = other.gradient_;
		Output_   = other.Output_  ;
	}

	Sigmoid& Sigmoid::operator=(const Sigmoid& other){
		X_        = other.X_       ;
		gradient_ = other.gradient_;
		Output_   = other.Output_  ;

		return *this;
	}

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
	//   Define: Relu Class
	// ===============================================================================

	Relu::Relu(){}
	Relu::~Relu(){}

	Relu::Relu(const MatrixXfR& x): Operation(x){}
	Relu::Relu(const Relu& other){
		X_ = other.X_;
		gradient_ = other.gradient_;
		Output_ = other.Output_;
	}

	Relu& Relu::operator=(const Relu& other){
		X_ = other.X_;
		gradient_ = other.gradient_;
		Output_ = other.Output_;

		return *this;
	}

	MatrixXfR Relu::forward(){
		apply_relu(X_, Output_);
		return Output_;
	}

	void Relu::backward(){
		gradient_ = MatrixXfR::Zero(Output_.rows(), Output_.cols());
		for(int i = 0; i < Output_.rows(); i++){
			for(int j = 0; j < Output_.cols(); j++){
				if(Output_(i, j) > 0){
					gradient_(i, j) = 1;
				}else{
					gradient_(i, j) = 0;
				}
			}
		}
	}

	string Relu::get_type(){
		return "Relu";
	}

	MatrixXfR Relu::get_gradient(){
		return gradient_;
	}

	// ===============================================================================
	//   Define: MeanSquaredError Class
	// ===============================================================================

	MeanSquaredError::MeanSquaredError(){};
	MeanSquaredError::~MeanSquaredError(){};
	MeanSquaredError::MeanSquaredError(const MatrixXfR &p, const MatrixXfR &t): Loss(p, t){};

	MeanSquaredError::MeanSquaredError(const MeanSquaredError& other){
		loss_       = other.loss_      ;
		prediction_ = other.prediction_;
		target_     = other.target_    ;
		gradient_   = other.gradient_  ;
	}

	MeanSquaredError& MeanSquaredError::operator=(const MeanSquaredError& other){
		loss_       = other.loss_      ;
		prediction_ = other.prediction_;
		target_     = other.target_    ;
		gradient_   = other.gradient_  ;

		return *this;
	}

	float MeanSquaredError::forward(MatrixXfR& P){
		float loss;
		MatrixXfR temp;

		prediction_ = P;

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

	string MeanSquaredError::get_type(){
		return "MSE";
	}
}
