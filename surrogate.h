#include <vector>
#include "gp.h"
#include <mutex>

#pragma once
using namespace std;

enum s_type { SEiso };

class Surrogate
{
  public:
    libgp::GaussianProcess *gp;
    
    //Functions
    Surrogate(int d, s_type t);
    void add(vector<double> x, double y);
    double _var(double x[]);
    double _mean(double x[]);
    void set_params(double, double);

  private:
    int dim;
    mutex mtx;
};

