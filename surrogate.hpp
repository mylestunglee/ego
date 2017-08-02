#pragma once
#include <vector>
#include <set>
#include "gp.h"

using namespace std;
using namespace libgp;

class Surrogate
{
  public:
    Surrogate(size_t dimension);
    ~Surrogate();
    void add(vector<double> x, double y);
    double mean(vector<double> x);
    double sd(vector<double> x);
	void optimise_space();
	double cross_validate();

  private:
	GaussianProcess* newGaussianProcess();
    void train();
	vector<double> extract_ys();

    size_t dimension;
	bool log_transform;
    GaussianProcess* gp;
	set<pair<vector<double>, double>> added;
	double added_mean;
	double added_sd;
};
