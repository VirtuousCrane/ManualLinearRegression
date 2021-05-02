#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include "src/utility.hpp"
#include "src/layers.hpp"

using namespace std;
using namespace Eigen;
using namespace utility;
using namespace layers;

double LEARNING_RATE = 0.01;

int main(){
// =======================================================
//  Defining the parameters
// =======================================================

	MatrixXfR input_matrix = load_csv("./src/dataset/boston/train.csv");
	normalize_cols(input_matrix);
	MatrixXfR target       = load_label("./src/dataset/boston/train_label.txt");
	MatrixXfR M1, O, P, L;
	float loss;

// =======================================================
//  Defining the Weight and Bias Gradient Matrices
// =======================================================

	MatrixXfR loss_gradient;
	MatrixXfR layer1_x_gradient, layer1_weight_gradient;
	MatrixXfR activation_gradient;
	MatrixXfR layer2_x_gradient, layer2_weight_gradient;
	MatrixXfR layer1_bias_gradient, layer2_bias_gradient;

	MatrixXfR layer1_W, layer1_B;
	MatrixXfR layer2_W, layer2_B;
	MatrixXfR Weight1, Weight2;
	MatrixXfR Bias1, Bias2;

// =======================================================
//  Defining the layers
// =======================================================

	FullyConnected   layer1, layer2  ;
	Sigmoid          activation_layer;
	MeanSquaredError Loss_Layer      ;

// =======================================================
//  Forward Pass
// =======================================================

	layer1 = FullyConnected(input_matrix, 13);
	M1     = layer1.forward();

	activation_layer = Sigmoid(M1);
	O = activation_layer.forward();

	layer2 = FullyConnected(O, 1);
	P = layer2.forward();

	Loss_Layer = MeanSquaredError(P, target);
	loss       = Loss_Layer.forward(P);

// =======================================================
//  Backward Pass
// =======================================================

	Loss_Layer.backward();

	layer2.backward();

	activation_layer.backward();

	layer1.backward();

// =======================================================
//  Training the layers en masse
// =======================================================

	double prev_loss = 0;
	int i = 0;
	while(i < 10000){
		// =======================================
		//  Forward Pass
		// =======================================

		M1 = layer1.forward();

		activation_layer.update(M1);
		O  = activation_layer.forward();

		layer2.update_input(O);
		P  = layer2.forward();

		loss = Loss_Layer.forward(P);

		// =======================================
		//  Backward Pass
		// =======================================

		Loss_Layer.backward()      ;
		layer2.backward()          ;
		activation_layer.backward();
		layer1.backward()          ;

		// =======================================
		//  Updating The Weights
		// =======================================

		loss_gradient = Loss_Layer.get_gradient();

		layer2_x_gradient      = layer2.get_x_gradient()   ;
		layer2_weight_gradient = layer2.get_gradient()     ;
		layer2_bias_gradient   = layer2.get_bias_gradient();

		activation_gradient = activation_layer.get_gradient();

		layer1_x_gradient      = layer1.get_x_gradient()   ;
		layer1_weight_gradient = layer1.get_gradient()     ;
		layer1_bias_gradient   = layer1.get_bias_gradient();

		layer2_W = layer2_weight_gradient * loss_gradient;
		layer2_B = loss_gradient.sum() * layer2_bias_gradient;

		layer2.update(layer2_W, layer2_B, LEARNING_RATE);

		layer1_W = loss_gradient * layer2_x_gradient;
		layer1_W = layer1_W.cwiseProduct(activation_gradient);
		layer1_W = layer1_weight_gradient * layer1_W;

		layer1_B = loss_gradient * layer2_x_gradient;
		layer1_B = layer1_B.cwiseProduct(activation_gradient);
		layer1_B = layer1_B.sum() * layer1_bias_gradient  ;

		layer1.update(layer1_W, layer1_B, LEARNING_RATE);

		cout << "Epoch: " << i++ << " Loss: " << loss << endl;

		// =======================================
		//  Updating the Learning Rate
		// =======================================

		if(i % 500 == 0){
			LEARNING_RATE /= 10;
			cout << "======================" << endl;
			cout <<  layer1.get_weight()(0) << endl;
			cout <<  layer2.get_weight()(0) << endl;
			cout << "======================" << endl;
		}
	}

// =======================================================
//  Testing the Neural Network
// =======================================================

	for(int i=0; i<10; i++){
		cout << "Predicted value: " << P(i, 0) << endl;
		cout << "Actual value: " << target(i, 0) << endl << endl;
	}

}
