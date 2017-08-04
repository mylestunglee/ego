#pragma once
#include <vector>
#include <set>
#include "gp.h"
#include "surrogate.hpp"

using namespace std;

class GaussianProcess : public Surrogate {
	public:
		GaussianProcess(size_t dimension);
		~GaussianProcess();
		void add(vector<double> x, double y);
		double mean(vector<double> x);
		double sd(vector<double> x);
		void optimise_space();
		double cross_validate();

	private:
		libgp::GaussianProcess* newGaussianProcess();
		void train();
		vector<double> extract_ys();
		vector<vector<double>> extract_xs();
		void update_whitener();
		vector<double> normalise_x(vector<double> x);

		size_t dimension;
		bool log_transform;
		libgp::GaussianProcess* gp;
		set<pair<vector<double>, double>> added;
		vector<double> x_mean;
		vector<double> x_sd;
		double y_mean;
		double y_sd;
};
