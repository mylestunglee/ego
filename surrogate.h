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
    void add(const vector<double> &x, double y, int cl);
    void add(const vector<double> &x, double y, int cl, int add);
    double var(double x[]);
    double mean(double x[]);
    pair<double, double> predict(double x[], bool raw=false);
    int svm_label(double x[]);
    void choose_svm_param(int num_folds, bool local=false);
    void train_gp(libgp::GaussianProcess *gp_, bool log_fit);
    void train_gp_first();
    double best_raw();
    double error();

    void train();
    bool is_trained;
    bool gp_is_trained;
    bool is_svm;
    vector<double> gamma;
    vector<double> C;
    double mean_fit;
    vector<vector<double>> training;
    vector<double> training_f;

  private:
    int dim;
    int num_train;
    int num_correct_class;
    bool use_log;
    vector<double> mean_data;
    vector<double> std_dev;
    //int elements;
    double best_raw_fit;
    mutex mtx;
    vector<vector<double>> add_training_svm;
    vector<double> add_training_cl;
    vector<struct svm_node *> training_svm_sparse;
    vector<double> training_cl;
    struct svm_node *s_node;
    struct svm_model *s_model;
    struct svm_parameter s_param;
    struct svm_problem s_prob;

};

