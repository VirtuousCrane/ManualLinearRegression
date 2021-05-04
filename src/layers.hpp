// Written By: Chayut Liewlom (63011134),
// the most lazy person in the class

#include <iostream>
#include "utility.hpp"
#include <Eigen/Dense>
#ifndef LAYERS_HPP
#define LAYERS_HPP

using namespace std;
using namespace Eigen;
using namespace utility;

namespace layers{
	class DifferentDimension{
		private:
			string errMessage;
		public:
			DifferentDimension();
			~DifferentDimension();
			string what();
	};

	class Operation{
		protected:
			MatrixXfR X_;
			MatrixXfR gradient_;
		public:
			Operation();
			Operation(const MatrixXfR& x);

			virtual MatrixXfR forward()  = 0;
			virtual void backward() = 0;
			virtual MatrixXfR get_gradient() = 0;
			virtual string get_type() = 0;

			virtual ~Operation();
	};

	class Loss{
		protected:
			float loss_;
			MatrixXfR prediction_, target_;
			MatrixXfR gradient_;
		public:
			Loss();
			Loss(const MatrixXfR& p, const MatrixXfR& t);

			virtual float forward(MatrixXfR& P) = 0;
			virtual void backward() = 0;
			virtual MatrixXfR get_gradient() = 0;
			virtual string get_type() = 0;

			virtual ~Loss();
	};

	class FullyConnected: public Operation{
		private:
			MatrixXfR Weight_;
			double Bias_;
			MatrixXfR Output_;
			MatrixXfR dPdB_, dPdX_;
		public:
			FullyConnected();
			FullyConnected(const MatrixXfR& x);
			FullyConnected(const MatrixXfR& x, int hidden_size);

			FullyConnected(const FullyConnected &other);
			FullyConnected& operator=(const FullyConnected& other);

			void set_weight(const MatrixXfR& weight);
			void set_bias(const double bias);

			MatrixXfR get_weight();
			double get_bias();

			string get_type();

			MatrixXfR forward();
			void backward();
			void update_input(const MatrixXfR& x);
			void update(const MatrixXfR& gradW, const MatrixXfR& gradB, double learning_rate);

			MatrixXfR get_gradient();
			MatrixXfR get_x_gradient();
			MatrixXfR get_bias_gradient();

			void get_weight_dimension();

			~FullyConnected();
	};

	class Sigmoid: public Operation{
		private:
			MatrixXfR Output_;
		public:
			Sigmoid();
			Sigmoid(const MatrixXfR& x);

			Sigmoid(const Sigmoid& other);
			Sigmoid& operator=(const Sigmoid& other);

			string get_type();

			MatrixXfR forward();
			void backward();
			void update(const MatrixXfR& x);

			MatrixXfR get_gradient();

			~Sigmoid();
	};

	class Relu: public Operation{
		private:
			MatrixXfR Output_;
		public:
		Relu();
		Relu(const MatrixXfR& x);

		Relu(const Relu& other);
		Relu& operator=(const Relu& other);

		string get_type();

		MatrixXfR forward();
		void backward();

		MatrixXfR get_gradient();

		~Relu();
	};

	class MeanSquaredError: public Loss{
		public:
			MeanSquaredError();
			MeanSquaredError(const MatrixXfR &p, const MatrixXfR &t);

			MeanSquaredError(const MeanSquaredError& other);
			MeanSquaredError& operator=(const MeanSquaredError& other);

			float forward(MatrixXfR& P);
			void backward();

			MatrixXfR get_gradient();
			string get_type();

			~MeanSquaredError();
	};
}
#endif
