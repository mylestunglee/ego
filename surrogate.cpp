#include "surrogate.h"
#include "gp.h"
#include <thread>
#include <iostream>
#include <sstream>

#define Malloc(type, n) (type *)malloc((n)*sizeof(type))

using namespace std;
using namespace libgp;

Surrogate::Surrogate(int d, s_type type, bool svm)
{
  dim = d;
  is_svm = svm;

  switch(type) {
    case SEiso:
      gp = make_shared<GaussianProcess>(dim, "CovSEiso");
      break;

    case SEard:
      gp = make_shared<GaussianProcess>(dim, "CovSEard");
      break;

    default:
      gp = make_shared<GaussianProcess>(dim, "CovSEard");
  }
  Eigen::VectorXd params(gp->covf().get_param_dim());
  //params.setZero();
  params << 1, 1;
  gp->covf().set_loghyper(params);

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

void Surrogate::choose_kernel(int num_folds)
{
  if(training.size() == 0) return;
  int subset_size = ceil(training.size() / num_folds);
  double best_mse = 10000000000;
  Eigen::VectorXd best_params;
  string best_cov = covs[0];
  for(auto cov : covs) {
    gp = make_shared<GaussianProcess>(dim, cov);
    Eigen::VectorXd params = Eigen::VectorXd(gp->covf().get_param_dim());
    params.setZero();
    gp->covf().set_loghyper(params);

    double mse = 0;
    for(int i = 0; i < num_folds; i++) {
      int fold = min(subset_size, (int) training.size() - i * subset_size);
      for(int j = 0; j < i * subset_size; j++) {
        gp->add_pattern(&(training[j][0]), training_f[j]);
      }
      for(int j = (i+1)*subset_size; j < training.size(); j++) {
        gp->add_pattern(&(training[j][0]), training_f[j]);
      }
      for(int j = i*subset_size; j < i*subset_size + fold; j++) {
        double mean_err = gp->f(&training[j][0]) - training_f[j];
	mse += mean_err * mean_err;
      }
    }
    if(mse < best_mse) {
      best_mse = mse;
      best_cov = cov;
      best_params = params;
    }
  }
}

//void Surrogate::set_params(double x, double y)
//{
//  Eigen::VectorXd params(gp->covf().get_param_dim());
//  params << x, y;
//  gp->covf().set_loghyper(params);
//}

void Surrogate::add(vector<double> x, double y)
{
  double *data = &x[0];
  gp->add_pattern(data, y);
  training.push_back(x);
  training_f.push_back(y);
}

void Surrogate::add(vector<double> x, double y, int cl)
{
  if(!is_svm) {
    cout << "Surrogate::add svm when not in svm mode" << endl;
    exit(-1);
  }
  add(x, y);
  training_cl.push_back(cl);
  is_trained = false;
}

void Surrogate::train()
{
  int amount = training.size();
  if(is_svm && !is_trained && amount > 0) {
    if(s_model != NULL) {
      svm_free_and_destroy_model(&s_model);
      free(s_node);
      free(s_prob.y);
      free(s_prob.x);
    }
    int elements = 0;
    for(int i = 0; i < amount; i++, elements++) {
      for(int k = 0; k < dim; k++) {
        if(training[i][k] != 0) elements++;
      }
    }
    s_prob.l = amount;
    s_prob.y = Malloc(double, amount);
    s_prob.x = Malloc(struct svm_node *, amount);
    s_node = Malloc(struct svm_node, elements);
    int k = 0;
    for(int i = 0, j = 0; i < training.size(); i++) {
      s_prob.x[i] = &s_node[j];
      s_prob.y[i] = training_cl[i];
      for(int k = 0; k < dim; k++) {
	if(training[i][k] != 0) {
          s_node[j].index = k;
	  s_node[j].value = training[i][k];
	  j++;
	}
      }
      k = max(k, j);
      s_node[j++].index = -1;
    }
    s_param.gamma = 1.0 / k;
    streambuf *old = cout.rdbuf(); // <-- save        
    stringstream ss;

    cout.rdbuf(ss.rdbuf());       // <-- redirect

    svm_check_parameter(&s_prob, &s_param);
    s_model = svm_train(&s_prob, &s_param);

    cout.rdbuf (old);              // <-- restore
    is_trained = true;
  }
}

pair<double, double> Surrogate::predict(double x[])
{
  if(svm_label(x) != 1) {
    return make_pair(100000000000, 0.0);
  } else {
    return make_pair(gp->f(x), gp->var(x));
  }
}


double Surrogate::var(double x[])
{
  double result =  gp->var(x);
  return result;
}

double Surrogate::mean(double x[]) 
{
  double result =  gp->f(x);
  return result;
}

int Surrogate::svm_label(double x[])
{
  if(!is_svm) return 1.0;
  if(!is_trained) {
    cout << "Haven't trained svm when calling, exiting" << endl;
    exit(-1);
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
}
