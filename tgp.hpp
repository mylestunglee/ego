#pragma once
#include "surrogate.hpp"
#include "gp.hpp"
#include <set>

using namespace std;

class TransferableGaussianProcess : public Surrogate {
	public:
		TransferableGaussianProcess(set<pair<vector<double>, double>> added);
		~TransferableGaussianProcess();
		void add(vector<double> x, double y);
		double mean(vector<double> x);
		double sd(vector<double> y);
		void optimise();
		double cross_validate();
		double cross_validate_parameter();
	private:
		set<pair<vector<double>, double>> added_new;
		set<pair<vector<double>, double>> added_old;
		GaussianProcess* transferred;
		GaussianProcess* parameter;

		void train();
};
