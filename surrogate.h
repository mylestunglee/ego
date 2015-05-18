#include <vector>
#include "gp.h"

#pragma once
using namespace std;

class Surrogate
{
  public:
    libgp::GaussianProcess *gp;
    vector<pair<vector<double>, double>> training;
    
    //Functions
    Surrogate(int d = 1);
    void add(double x[], double y);
    double _var(double x[]);
    double _mean(double x[]);
    double y_best();

  private:
    int dim;
    double _y_best;
};
