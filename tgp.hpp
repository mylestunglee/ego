#pragma once
#include "gp.hpp"
#include <set>

using namespace std;

class TransferredGaussianProcess {
	public:
		TransferredGaussianProcess(set<pair<vector<double>, double>> added);
		~TransferredGaussianProcess();
		void add(vector<double> x, double y);
		double mean(vector<double> x);
		double sd(vector<double> y);
		void optimise();
		double cross_validate();
	private:
		set<pair<vector<double>, double>> added_new;
		set<pair<vector<double>, double>> added_old;
		GaussianProcess* transferred;

		void train();
};
