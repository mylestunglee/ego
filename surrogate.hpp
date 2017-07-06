#include <vector>
#include "gp.h"
#include "libsvm-3.20/svm.h"
#include <mutex>

#pragma once
using namespace std;

enum s_type {SEiso, SEard};

class Surrogate
{
  public:
    libgp::GaussianProcess *gp;

    //Functions
    Surrogate(int d, s_type t, bool svm=false, bool use_log=false);
    ~Surrogate();
    void add(const vector<double> &x, double y);
    double var(double x[]);
    double mean(double x[]);
    pair<double, double> predict(double x[]);
    void choose_svm_param(int num_folds, bool local=false);
    void train_gp(libgp::GaussianProcess *gp_, bool log_fit);
    void train_gp_first();
    double best_raw();
    double error();

    void train();

  private:
    int dim;
};
