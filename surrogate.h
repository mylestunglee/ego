#include <vector>
#include "gp.h"
#include <mutex>

#pragma once
using namespace std;

class Surrogate
{
  public:
    libgp::GaussianProcess *gp;
    
    //Functions
    Surrogate(int d = 1);
    void add(vector<double> x, double y);
    double _var(double x[]);
    double _mean(double x[]);

  private:
    int dim;
    mutex mtx;
};
