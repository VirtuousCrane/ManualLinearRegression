#include <iostream>
#include "utility.hpp"
#include <Eigen/Dense>
#ifndef LAYERS_HPP
#define LAYERS_HPP

using namespace std;
using namespace Eigen;
using namespace utility;

namespace layers{
	MatrixXfR init_weights(int X_size, int hidden_layer_size);
	MatrixXfR init_bias(int hidden_size);

	class DifferentDimension{
		private:
			string errMessage;
		public:
			DifferentDimension();
			string what();
	};

	class Operation{
		protected:
			MatrixXfR X_;
			MatrixXfR gradient_;
		public:
			Operation(const MatrixXfR& x);

			virtual MatrixXfR forward()  = 0;
			virtual void backward() = 0;
			virtual MatrixXfR get_gradient() = 0;
			virtual string get_type() = 0;
	};

	class Loss{
		protected:
			float loss_;
			MatrixXfR prediction_, target_;
			MatrixXfR gradient_;
		public:
			Loss(const MatrixXfR& p, const MatrixXfR& t);

			virtual float forward() = 0;
			virtual void backward() = 0;
			virtual MatrixXfR get_gradient() = 0;
			virtual string get_type() = 0;
	};

	class FullyConnected: public Operation{
		private:
			MatrixXfR Weight_;
			MatrixXfR Bias_;
			MatrixXfR Output_;
			MatrixXfR dPdB_;
		public:
			FullyConnected(const MatrixXfR& x, int hidden_size);

			void set_weight(const MatrixXfR& weight);
			void set_bias(const MatrixXfR& bias);

			string get_type();

			MatrixXfR forward();
			void backward();

			MatrixXfR get_gradient();
			MatrixXfR get_bias_gradient();
	};

	class Sigmoid: public Operation{
		private:
			MatrixXfR Output_;
		public:
			Sigmoid(const MatrixXfR& x);

			string get_type();

			MatrixXfR forward();

			void backward();
			MatrixXfR get_gradient();
	};

	class MeanSquaredError: public Loss{
		MeanSquaredError(const MatrixXfR &p, const MatrixXfR &t);

		float forward();
		void backward();

		MatrixXfR get_gradient();
		string get_type();
	};
}
#endif
