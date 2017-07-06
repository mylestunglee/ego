#pragma once
#include <vector>
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
    void train();

  private:
    size_t dimension;
    GaussianProcess* gp;
};
