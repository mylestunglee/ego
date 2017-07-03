#include "surrogate.hpp"
#include "gp.h"
#include "cg.h"
#include <thread>
#include <iostream>
#include <sstream>

#define Malloc(type, n) (type *)malloc((n)*sizeof(type))

using namespace std;
using namespace libgp;

Surrogate::Surrogate(int d, s_type type, bool svm, bool log_b)
{
  dim = d;
  num_train = 0;
  //num_train_svm = 0;
  num_correct_class = 0;
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
  mean_data.resize(dim, 0.0);
  std_dev.resize(dim, 1.0);
  use_log = log_b;

  s_node = NULL;
  s_model = NULL;
  is_trained = false;
  gp_is_trained = false;

  for(int i = 0; i < dim; i++) {
    mean_data[i] = 0;
    std_dev[i] = 0;
    mean_fit = 0;
  }

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

void Surrogate::choose_svm_param(int num_folds, bool local)
{
  //if(s_model != NULL) {
  //  svm_free_and_destroy_model(&s_model);
  //  free(s_node);
  //  free(s_prob.y);
  //  free(s_prob.x);
  //}

  //int elements = 0;
  //int amount = training_svm.size();
  //for(int i = 0; i < amount; i++, elements++) {
  //  for(int k = 0; k < dim; k++) {
  //    if(training_svm[i][k] != 0) elements++;
  //  }
  //}

  //s_prob.l = amount;
  //s_prob.y = Malloc(double, amount);
  //s_prob.x = Malloc(struct svm_node *, amount);
  //s_node = Malloc(struct svm_node, elements);

  //for(int i = 0, j = 0; i < amount; i++) {
  //  s_prob.x[i] = &s_node[j];
  //  s_prob.y[i] = training_cl[i];
  //  for(int k = 0; k < dim; k++) {
  //    if(training_svm[i][k] != 0) {
  //      s_node[j].index = k;
  //      s_node[j].value = training_svm[i][k];
  //      j++;
  //    }
  //  }
  //  s_node[j++].index = -1;
  //}

  if(s_model != NULL) {
    svm_free_and_destroy_model(&s_model);
  }

  //Add to SVM problem
  for(size_t i = 0; i < add_training_svm.size(); i++) {
    vector<struct svm_node> *node = new vector<svm_node>();
    for(int k = 0; k < dim; k++) {
      if(add_training_svm[i][k] != 0) {
        struct svm_node n;
        n.index = k;
        n.value = add_training_svm[i][k] - mean_data[k];
	if(std_dev[k] != 0) {
          n.value /= std_dev[k];
	}
        node->push_back(n);
      }
    }
    struct svm_node n;
    n.index = -1;
    node->push_back(n);
    struct svm_node *p = &(*node)[0];
    training_svm_sparse.push_back(p);
    training_cl.push_back(add_training_cl[i]);
  }
  add_training_svm.clear();
  add_training_cl.clear();
  s_prob.x = (struct svm_node **) &training_svm_sparse[0];
  s_prob.y = &training_cl[0];
  s_prob.l = (int) training_svm_sparse.size();

  double *target = Malloc(double, s_prob.l);
  int best = 0;
  double best_gamma, best_C;
  s_param.nr_weight = 2;
  if(s_param.weight == NULL) {
    s_param.weight_label = Malloc(int, 2);
    s_param.weight = Malloc(double, 2);
  }
  for(int i = 0; i < 2; i++) s_param.weight_label[i] = i+1;
  s_param.weight[0] = s_prob.l - num_correct_class;
  s_param.weight[1] = s_prob.l - s_param.weight[0];
  //Weight so if we only have a couple okay points we don't want to misclassify them
  if(s_param.weight[0] / (double) s_prob.l < 0.5) {
    s_param.weight[0] = 1;
    s_param.weight[1] = 1;
  }

  if(gamma.size() > 0 && C.size() > 0 && num_correct_class > 1) {
    cout << "Performing " << num_folds << " fold validation for SVM" << endl;
    int fold = min(num_folds, num_correct_class);
    for(size_t i = 0; i < gamma.size(); i++) {
      for(size_t j = 0; j < C.size(); j++) {
        int curr = 0;
        s_param.gamma = gamma[i];
        s_param.C = C[j];
        svm_check_parameter(&s_prob, &s_param);
        svm_cross_validation(&s_prob, &s_param, fold, target);
        for(int k = 0; k < s_prob.l; k++) {
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
  // cout << "Optimised SVM, gamma: " <<s_param.gamma<< " C: " << s_param.C << endl;

  free(target);

  svm_check_parameter(&s_prob, &s_param);

  //Best parameters
  s_model = svm_train(&s_prob, &s_param);

}

void Surrogate::add(const vector<double> &x, double y)
{
  training.push_back(x);
  training_f.push_back(y);
  num_train++;
}

void Surrogate::add(const vector<double> &x, double y, int cl)
{
  if(!is_svm) {
    cout << "Surrogate::add svm when not in svm mode" << endl;
    exit(-1);
  }
  add(x, y);
  training_cl.push_back(cl);
  //num_train_svm++;
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

  add_training_svm.push_back(x);
  add_training_cl.push_back(cl);
  if(cl == 1) num_correct_class++;
}

void Surrogate::train_gp(libgp::GaussianProcess *gp_, bool log_fit)
{
  if(num_train == 0) return;
  best_raw_fit = 1000000.0;
  int amount = training.size();
  for(int i = 0; i < dim; i++) {
    mean_data[i] = 0;
    std_dev[i] = 0;
    mean_fit = 0;
  }
  for(int i = 0; i < amount; i++) {
    for(int j = 0; j < dim; j++) {
      mean_data[j] += training[i][j];
    }
    if(log_fit) {
      mean_fit += log(training_f[i]);
    } else {
      mean_fit += training_f[i];
    }
  }
  mean_fit /= amount;
  for(int i = 0; i < dim; i++) {
    mean_data[i] /= amount;
  }
  for(int i = 0; i < amount; i++) {
    for(int j = 0; j < dim; j++) {
      std_dev[j] += pow(training[i][j] - mean_data[j], 2);
    }
  }
  for(int i = 0; i < dim; i++) {
    std_dev[i] = std_dev[i] / (amount - 1);
    std_dev[i] = sqrt(std_dev[i]);
  }
  for(int i = amount - num_train; i < amount; i++) {
    double data[dim];
    for(int j = 0; j < dim; j++) {
      data[j] = (training[i][j] - mean_data[j]);
      if(std_dev[j] != 0) {
        data[j] /= std_dev[j];
      }
    }
    double fit = training_f[i] - mean_fit;
    if(log_fit) {
      fit = log(training_f[i]) - mean_fit;
    }
    if(fit < best_raw_fit) {
      best_raw_fit = fit;
    }
    gp_->add_pattern(data, fit);
  }
  Eigen::VectorXd params(gp->covf().get_param_dim());
  params.setZero();
  gp_->covf().set_loghyper(params);
  CG cg;
  cg.maximize(gp_, 50, 0);
  num_train = 0;
  gp_is_trained = true;
}

void Surrogate::train_gp_first()
{
  if(gp) delete gp;
  num_train = training.size();
  gp = new GaussianProcess(dim, "CovSum(CovSEard, CovNoise)");
  train_gp(gp, false);
  long double ll = gp->log_likelihood();

  num_train = training.size();
  libgp::GaussianProcess* gp_log = new GaussianProcess(dim, "CovSum(CovSEard, CovNoise)");
  train_gp(gp_log, true);
  long double log_ll = gp_log->log_likelihood();
  if(log_ll > ll) {
    cout << "Defaulted to using log on GP";
    cout << " log like: " << log_ll;
    cout << " normal like: " << ll << endl;
    use_log = true;
    delete gp;
    gp = gp_log;
  } else {
    cout << "Defaulted to not using log on GP";
    cout << " log like: " << log_ll;
    cout << " normal like: " << ll << endl;
    use_log = false;
    delete gp_log;
  }
}

void Surrogate::train()
{
  train_gp(gp, use_log);
  if(is_svm) choose_svm_param(10);
  is_trained = true;
}

pair<double, double> Surrogate::predict(double x[], bool raw)
{
  if(is_trained) {
    double data[dim];
    for(int j = 0; j < dim; j++) {
      data[j] = (x[j] - mean_data[j]);
      if(std_dev[j] != 0) {
        data[j] /= std_dev[j];
      }
    }
    //cout << "Normalised" << endl;
    //cout << "label =" << svm_label(data) << endl;
    if(svm_label(data) == 1) {
      if(raw) {
          return make_pair(gp->f(data), gp->var(data));
      } else {
        if(use_log) {
          return make_pair(exp(mean_fit + gp->f(data)), gp->var(data));
        } else {
          return make_pair(mean_fit + gp->f(data), gp->var(data));
        }
      }
    }
  }
  return make_pair(100000000000.0, 0.0);
}

double Surrogate::best_raw()
{
  return best_raw_fit;
}

double Surrogate::error()
{
  Eigen::VectorXd params = gp->covf().get_loghyper();
  double noise = params(gp->covf().get_param_dim() - 1);
  noise = exp(noise);
  return noise;
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
      node[j].value = x[i] - mean_data[i];
      if(std_dev[i] != 0) {
        node[j].value /= std_dev[i];
      }
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
    //free(s_node);
    //free(s_prob.y);
    //free(s_prob.x);
    for(size_t i = 0; i < training_svm_sparse.size(); i++) {
      delete(training_svm_sparse[i]);
    }
  }
  delete gp;
}
