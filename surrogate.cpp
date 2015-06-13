#include "surrogate.h"
#include "gp.h"
#include "cg.h"
#include <thread>
#include <iostream>
#include <sstream>

#define Malloc(type, n) (type *)malloc((n)*sizeof(type))

using namespace std;
using namespace libgp;

Surrogate::Surrogate(int d, s_type type, bool svm)
{
  dim = d;
  amount_to_train = 0;
  amount_correct_class = 0;
  is_svm = svm;

  switch(type) {
    case SEiso:
      gp = new GaussianProcess(dim, "CovSum(CovSEiso, CovNoise)");
      break;

    case SEard:
      gp = new GaussianProcess(dim, "CovSum(CovSEard, CovNoise)");
      break;

    default:
      gp = new GaussianProcess(dim, "CovSEard");
  }
  Eigen::VectorXd params(gp->covf().get_param_dim());
  for(size_t i = 0; i < gp->covf().get_param_dim(); i++) {
    params(i) = -1;
  }
  gp->covf().set_loghyper(params);

  s_node = NULL;
  s_model = NULL;
  is_trained = false;
  covs = {"CovSEiso", "CovSEard", "CovSum (CovSEiso, CovNoise)"};
  if(is_svm) {
    s_param.svm_type = C_SVC;
    s_param.kernel_type = RBF;
    s_param.degree = 3;
    s_param.gamma = 1.0 / dim;
    s_param.coef0 = 0;
    s_param.nu = 0.5;
    s_param.cache_size = 100;
    s_param.C = 1;
    s_param.eps = 1e-3;
    s_param.p = 0.1;
    s_param.shrinking = 1;
    s_param.probability = 0;
    s_param.nr_weight = 0;
    s_param.weight_label = NULL;
    s_param.weight = NULL;
    s_prob.y = NULL;
    s_prob.x = NULL;
  }
}

//double Surrogate::gp_mean_square(const vector<double> pam)
//{
//  int folds = 5;
//  if(gp) delete gp;
//  gp = new GaussianProcess(dim, "CovSum(CovSEard, CovNoise)");
//  Eigen::VectorXd params(gp->covf().get_param_dim());
//  for(size_t i = 0; i < pam.size(); i++) {
//    params(i) = pam[i];
//  }
//  gp->covf().set_loghyper(params);
//  int batch = ceil(training.size() / folds);
//  double mse = 0.0;
//  int size = training.size();
//  for(int i = 0; i < folds; i++) {
//    for(int j = 0; j < i * batch; j++) {
//      gp->add_pattern(&training[j][0], training_f[j]);
//    }
//    for(int j = (i+1)*batch; j < size; j++) {
//      gp->add_pattern(&training[j][0], training_f[j]);
//    }
//    for(int j = i*batch; j < min((i+1)*batch, size); j++) {
//      double pred = gp->f(&training[j][0]) - training_f[j];
//      mse += pred * pred;
//    }
//  }
//  delete gp;
//  return mse;
//}
//
//void Surrogate::choose_gp_params(int num_folds)
//{
//  vector<double> params(dim+2, 1.0);
//  int loops = 0;
//  int loop[dim+1];
//  unsigned long long npts_plus[dimension + 2];
//  npts_plus[dimension] = 1;
//  double best_mse = 1000000000000;
//  for(double i = 0.001; i < 10.0; i = min(i*10, i+1)) {
//    for(double i = 0.001; i < 10.0; i = min(i*10, i+1)) {
//      for(double i = 0.001; i < 10.0; i = min(i*10, i+1)) {
//      }
//    }
//  }
//}

void Surrogate::choose_svm_param(int num_folds, bool local)
{
  vector<double> gamma, C;
  if(local) {
    for(double i = -10.; i < 31; i++) {
      gamma.push_back(10 * pow(1.25, i));
    }
    for(double i = -30.; i < 31; i++) {
      C.push_back(pow(1.5, i));
    }
  } else {
    for(double i = -20.; i < 21; i++) {
      gamma.push_back(pow(1.2, i));
      C.push_back(10 * pow(1.25, i));
    }
  }

  if(s_model != NULL) {
    svm_free_and_destroy_model(&s_model);
    free(s_node);
    free(s_prob.y);
    free(s_prob.x);
  }

  int elements = 0;
  int amount = training_svm.size();
  for(int i = 0; i < amount; i++, elements++) {
    for(int k = 0; k < dim; k++) {
      if(training_svm[i][k] != 0) elements++;
    }
  }

  s_prob.l = amount;
  s_prob.y = Malloc(double, amount);
  s_prob.x = Malloc(struct svm_node *, amount);
  s_node = Malloc(struct svm_node, elements);

  for(int i = 0, j = 0; i < amount; i++) {
    s_prob.x[i] = &s_node[j];
    s_prob.y[i] = training_cl[i];
    for(int k = 0; k < dim; k++) {
      if(training_svm[i][k] != 0) {
        s_node[j].index = k;
        s_node[j].value = training_svm[i][k];
        j++;
      }
    }
    s_node[j++].index = -1;
  }


  double *target = Malloc(double, amount);
  int best = 0;
  double best_gamma, best_C;
  s_param.nr_weight = 2;
  if(!s_param.weight) {
    s_param.weight_label = Malloc(int, 2);
    s_param.weight = Malloc(double, 2);
  }
  for(int i = 0; i < 2; i++) s_param.weight_label[i] = i+1;
  s_param.weight[0] = amount - amount_correct_class;
  s_param.weight[1] = amount - s_param.weight[0];
  if(s_param.weight[0] / (double) amount > 0.5) {
    s_param.weight[0] = 1;
    s_param.weight[1] = 1;
  }

  if(amount_correct_class > 1) {
    int folds = min(amount_correct_class, num_folds);
    for(size_t i = 0; i < gamma.size(); i++) {
      for(size_t j = 0; j < C.size(); j++) {
        int curr = 0;
        s_param.gamma = gamma[i];
        s_param.C = C[j];
        svm_check_parameter(&s_prob, &s_param);
        svm_cross_validation(&s_prob, &s_param, folds, target);
        for(int k = 0; k < amount; k++) {
          curr += (int) (target[k] == training_cl[k]);
        }
        if(curr > best) {
          best = curr;
          best_gamma = gamma[i];
          best_C = C[i];
        }
      }
    }
    s_param.gamma = best_gamma;
    s_param.C = best_C;
  } else {
    s_param.gamma = 0.05;
    s_param.C = 10.0;
  }
  free(target);

  svm_check_parameter(&s_prob, &s_param);

  //Best parameters
  s_model = svm_train(&s_prob, &s_param);

}

void Surrogate::add(const vector<double> &x, double y)
{
  training.push_back(x);
  training_f.push_back(y);
  amount_to_train++;
}

void Surrogate::add(const vector<double> &x, double y, int cl)
{
  if(!is_svm) {
    cout << "Surrogate::add svm when not in svm mode" << endl;
    exit(-1);
  }
  add(x, y);
  training_cl.push_back(cl);
}

void Surrogate::add(const vector<double> &x, double y, int cl, int addReturn)
{
  if(!is_svm) {
    cout << "Surrogate::addReturn svm with addReturnReturn when not in svm mode" << endl;
    exit(-1);
  }
  //Have to addReturn this in case no fitness result produced
  if(addReturn == 0) {
    add(x, y);
  }
  training_svm.push_back(x);
  training_cl.push_back(cl);
  if(cl == 1) amount_correct_class++;
}

void Surrogate::train()
{
  int amount = training.size();
  for(int i = amount - amount_to_train; i < amount; i++) {
    double *data = &training[i][0];
    gp->add_pattern(data, training_f[i]);
  }
  amount_to_train = 0;
  CG cg;
  cg.maximize(gp, 50, 0);
  if(is_svm) choose_svm_param(5);
  is_trained = true;
}

pair<double, double> Surrogate::predict(double x[])
{
  if(svm_label(x) != 1) {
    return make_pair(100000000000.0, 0.0);
  } else {
    return make_pair(gp->f(x), gp->var(x));
  }
}


double Surrogate::var(double x[])
{
  return gp->var(x);
}

double Surrogate::mean(double x[]) 
{
  return gp->f(x);
}

int Surrogate::svm_label(double x[])
{
  if(!is_svm) return 1;
  if(!s_model) {
    return 1;
  }
  struct svm_node *node = Malloc(struct svm_node, dim+1);
  int j = 0;
  for(int i = 0; i < dim; i++) {
    if(x[i] != 0) {
      node[j].index = i;
      node[j].value = x[i];
      j++;
    }
  }
  node[j].index = -1;
  int result = round(svm_predict(s_model, node));
  free(node);
  return result;
}

Surrogate::~Surrogate()
{
  if(s_model != NULL) {
    svm_free_and_destroy_model(&s_model);
    free(s_node);
    free(s_prob.y);
    free(s_prob.x);
  }
  delete gp;
}
