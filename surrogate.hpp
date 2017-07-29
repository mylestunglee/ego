#pragma once
#include <vector>
#include <set>
#include "gp.h"

using namespace std;
using namespace libgp;

class Surrogate
{
  public:
    Surrogate(size_t dimension, bool log_transform = false,
		bool fixed_space = true);
    ~Surrogate();
    void add(vector<double> x, double y);
    double mean(vector<double> x);
    double sd(vector<double> x);
    void train();
	void optimise_space();
	double cross_validate();

  private:
	GaussianProcess* newGaussianProcess();

    size_t dimension;
	bool log_transform;
	bool fixed_space;
	bool trained;
    GaussianProcess* gp;
	set<pair<vector<double>, double>> added;
};
