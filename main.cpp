#include <iostream>
#include <vector>
#include <limits>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include "src/utility.hpp"
#include "src/layers.hpp"

using namespace std;
using namespace Eigen;
using namespace utility;
using namespace layers;

double LEARNING_RATE = 0.01;

class Evaluate{
	public:
		static void quit(){
			exit(0);
		}

		static void predict(vector<float>& norm, FullyConnected& layer1, FullyConnected& layer2, Sigmoid& activation_layer){
			float crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, black, lstat;
			MatrixXfR M1, O, P;

			cout << "Enter per capita crime rate: ";
			cin  >> crim;

			cout << "Enter residential land zoned for lots over 25000 sqft: ";
			cin  >> zn;

			cout << "Enter proportion of non-retail business acres per town: ";
			cin  >> indus;

			cout << "Enter 1 if tract bounds Charles river, 0 if not: ";
			cin  >> chas;

			cout << "Enter nitrogen oxide concentration (ppm): ";
			cin  >> nox;

			cout << "Enter average number of rooms per dwelling: ";
			cin  >> rm;

			cout << "Enter proportion of owner-occupied units built prior to 1940: ";
			cin  >> age;

			cout << "Enter weighted mean of distances to five Boston employment centres: ";
			cin  >> dis;

			cout << "Enter index of accessibility to radial highways: ";
			cin  >> rad;

			cout << "Enter property tax rate per $10,000: ";
			cin  >> tax;

			cout << "Enter pupil-teacher ratio by town: ";
			cin  >> ptratio;

			cout << "Enter proportion of blacks by town: ";
			cin  >> black;

			cout << "Enter lower status of the population: ";
			cin  >> lstat;

			MatrixXfR inp;
			inp.resize(1, 13);

			inp << crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, black, lstat;

			for(size_t i=0; i<inp.cols(); i++){
				inp(0, i) /= norm[i];
			}

			layer1.update_input(inp);
			M1 = layer1.forward();

			activation_layer.update(M1);
			O = activation_layer.forward();

			layer2.update_input(O);
			P = layer2.forward();

			cout << "Predicted Value: " << P(0, 0) << endl;

		}
};


int main(){
// =======================================================
//  Defining the parameters
// =======================================================

	MatrixXfR input_matrix = load_csv("./src/dataset/boston/train.csv");
	normalize_cols(input_matrix);
	MatrixXfR target       = load_label("./src/dataset/boston/train_label.txt");
	MatrixXfR M1, O, P, L;
	float loss;
	unsigned int epoch = 10000;

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
	double Bias1, Bias2;

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
//  To train, or not to train
// =======================================================
	int c;
	while(1){
		cout << "============================================" << endl;
		cout << " What would you like to do? "                 << endl;
		cout << " 1. Train The Model"                          << endl;
		cout << " 2. Load A Previously Trained Model "         << endl;
		cout << " 0. Exit "                                    << endl;
		cout << "============================================" << endl;
		cin  >> c;

		if(!cin){

			cin.clear();
			cin.ignore(numeric_limits<streamsize>::max(), '\n');
			cout << "Invalid Choice" << endl;
			continue;

		} else if (c == 2) {

			try{

				// Loading data
				Weight1 = load_csv("./src/model/weight1.csv");
				Weight2 = load_csv("./src/model/weight2.csv");
				cout << "Loaded weight" << endl;
				Bias1 = load_csv("./src/model/bias1.csv")(0, 0);
				Bias2 = load_csv("./src/model/bias2.csv")(0, 0);

				layer1.set_weight(Weight1);
				layer1.set_bias(Bias1);

				layer2.set_weight(Weight2);
				layer2.set_bias(Bias2);

				// Forward Pass
				M1     = layer1.forward();

				activation_layer.update(M1);
				O = activation_layer.forward();

				layer2.update_input(M1);
				P = layer2.forward();

				loss       = Loss_Layer.forward(P);

			} catch(...) {

				cout << "Invalid Input" << endl;
				continue;

			}

			string ans;
			cout << "Would you like to continue training your model? Y/N" << endl;
			cin  >> ans;

			if(ans == "Y" || ans == "y"){

				c = 1;

			} else if (ans == "N" || ans == "n"){

				break;

			} else {

				cout << "Invalid Input" << endl;
				continue;

			}

		}

		if (c == 1) {

			int temp = -1;
			cout << "Please enter the number of epochs you want to train for: ";
			cin  >> temp;

			if(!cin || temp <= 0){

				cin.clear();
				cin.ignore(numeric_limits<streamsize>::max(), '\n');
				cout << "Invalid input" << endl;
				continue;

			}

			epoch = temp;
			temp = 0;

			cout << "Please enter your learning rate (this learning rate will decrease by 1/10 every 500 epochs for the first 3000 epochs)" << endl;
			cout << "Enter 0 for default: ";
			cin  >> temp;

			if(!cin || temp < 0){

				cin.clear();
				cin.ignore(numeric_limits<streamsize>::max(), '\n');
				cout << "Invalid input" << endl;

			}

			if(temp != 0){
				LEARNING_RATE = temp;
			}

			break;

		} else if (c == 0) {

			cout << "Why did you even come here...?" << endl;
			return 0;

		}
	}

// =======================================================
//  Training the layers en masse
// =======================================================

	double prev_loss = 0;
	int i = 0;
	while(i < epoch && c == 1){
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
			if(i < 3000){
				LEARNING_RATE /= 10;
			}
			cout << "============================================" << endl;
			cout << "Layer 1 Weight at (0, 0): "      << layer1.get_weight()(0) << endl;
			cout << "Layer 1 Bias at (0, 0): "      << layer1.get_bias() << endl;
			cout << "Layer 2 Weight at (0, 0): "      << layer2.get_weight()(0) << endl;
			cout << "Layer 2 Bias at (0, 0): "      << layer2.get_bias() << endl;
			cout << "Activation gradient at (0, 0): " << activation_layer.get_gradient()(0, 0) << endl;
			cout << "============================================" << endl;
		}
	}

// =======================================================
//  Testing the Neural Network
// =======================================================
/*
	for(int i=0; i<10; i++){
		cout << "Predicted value: " << P(i, 0) << endl;
		cout << "Actual value: " << target(i, 0) << endl << endl;
	}

	input_matrix = load_csv("./src/dataset/boston/train.csv");
	input_matrix /= norm;
	target     = load_label("./src/dataset/boston/train_label.txt");

	layer1.update_input(input_matrix);
	M1 = layer1.forward();

	activation_layer.update(M1);
	O  = activation_layer.forward();

	layer2.update_input(O);
	P  = layer2.forward();

	for(int i=0; i<10; i++){
		cout << "Predicted value: " << P(i, 0) << endl;
		cout << "Actual value: " << target(i, 0) << endl << endl;
	}

*/

	cout << "============================================" << endl;
	cout << " Info about this Linear Regression Model: "   << endl;
	cout << " No. of hidden layers: 1"                     << endl;
	cout << " Loss (MSE): " << loss                        << endl;
	cout << " RMSE: " << sqrt(loss)                        << endl;
	cout << "============================================" << endl;

// =======================================================
//  Get user input data
// =======================================================

	vector<float> norm;

	for(int i=0; i<input_matrix.cols(); i++){
		norm.push_back(input_matrix.col(i).norm());
	}

	int choice;

	cin.clear();
	cin.ignore(numeric_limits<streamsize>::max(), '\n');

	while(1){
		cout << "============================================" << endl;
		cout << "Choose What you want to do: " << endl;
		cout << "1. Predict"                   << endl;
		cout << "2. Save Model"                << endl;
		cout << "0. Exit"                      << endl;
		cout << "============================================" << endl;

		cin >> choice;

		if(!cin){
			cin.clear();
			cin.ignore(numeric_limits<streamsize>::max(), '\n');
			continue;
		}

		if(choice == 1){
			Evaluate::predict(norm, layer1, layer2, activation_layer);
		} else if (choice == 2) {
			Weight1 = layer1.get_weight();
			Weight2 = layer2.get_weight();
			utility::write_to_file("src/model/weight1.csv", Weight1);
			utility::write_to_file("src/model/weight2.csv", Weight2);
			utility::write_to_file("src/model/bias1.csv", layer1.get_bias());
			utility::write_to_file("src/model/bias2.csv", layer2.get_bias());
			continue;
		} else if (choice == 0) {
			break;
		} else {
			cout << "Incorrect Choice, Please Enter Again" << endl;
		}
	}
}
