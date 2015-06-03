#include <vector>
#include "gp.h"
#include "libsvm-3.20/svm.h"
#include <mutex>

#pragma once
using namespace std;

enum s_type { SEiso };

class Surrogate
{
  public:
    libgp::GaussianProcess *gp;
    
    //Functions
    Surrogate(int d, s_type t, bool svm=false);
    void add(vector<double> x, double y);
    void add(vector<double> x, double y, int cl);
    double var(double x[]);
    double mean(double x[]);
    pair<double, double> predict(double x[]);
    double svm_label(double x[]);

    void set_params(double, double);
    void train();

  private:
    int dim;
    mutex mtx;
    bool is_svm;
    bool is_trained = false;
    vector<vector<double>> training;
    vector<int> training_cl;
    struct svm_node *s_node;
    struct svm_model *s_model;
    struct svm_parameter s_param;
    struct svm_problem s_prob;

};

