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
    Surrogate(int d, s_type t, bool svm=false);
    ~Surrogate();
    void add(vector<double> x, double y);
    void add(vector<double> x, double y, int cl);
    double var(double x[]);
    double mean(double x[]);
    pair<double, double> predict(double x[]);
    int svm_label(double x[]);

    void choose_kernel(int folds);
    void train();
    bool is_trained;

  private:
    int dim;
    mutex mtx;
    bool is_svm;
    vector<vector<double>> training;
    vector<double> training_f;
    vector<int> training_cl;
    struct svm_node *s_node;
    struct svm_model *s_model;
    struct svm_parameter s_param;
    struct svm_problem s_prob;
    vector<string> covs;

};

