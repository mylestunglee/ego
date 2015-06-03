#include "surrogate.h"
#include <dlib/svm.h>
#include "gp.h"

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

using namespace std;
using namespace libgp;

Surrogate::Surrogate(int d, s_type type, bool svm)
{
  dim = d;

  if(type == SEiso) {
    gp = new GaussianProcess(dim, "CovSEiso");
  } 

  if(svm) {
    is_svm = true;
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

void Surrogate::set_params(double x, double y)
{
  Eigen::VectorXd params(gp->covf().get_param_dim());
  params << x, y;
  gp->covf().set_loghyper(params);
}

void Surrogate::add(vector<double> x, double y)
{
  double *data = &x[0];
  gp->add_pattern(data, y);
}

void Surrogate::add(vector<double> x, double y, int cl)
{
  if(!is_svm) {
    cout << "Surrogate::add svm when not in svm mode" << endl;
    exit(-1);
  }
  add(x, y);
  training.push_back(x);
  training_cl.push_back(cl);
  is_trained = false;
}
void Surrogate::train()
{
  if(!is_trained) {
    int amount = training.size();
    int elements = amount * dim;
    if(s_model) {
      svm_free_and_destroy_model(&s_model);
      free(s_node);
      free(s_prob.y);
      free(s_prob.x);
    }
    s_prob.l = amount;
    s_prob.y = Malloc(double, amount);
    s_prob.x = Malloc(struct svm_node *, amount);
    s_node = Malloc(struct svm_node, elements);
    for(int i = 0; i < training.size(); i++) {
      s_prob.x[i] = &s_node[i*(dim+1)];
      s_prob.y[i] = training_cl[i];
      for(int k = 0; k < dim; k++) {
        s_node[i*(dim+1) + k].index = k;
	s_node[i*(dim+1) + k].value = training[i][k];
      }
      s_node[(i+1)*(dim+1)].index = -1;
    }
    s_model = svm_train(&s_prob, &s_param);
    is_trained = true;
  }
}
pair<double, double> Surrogate::predict(double x[])
{
  if(svm_label(x) != 1.0) {
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

double Surrogate::svm_label(double x[])
{
  if(!is_svm) return 1.0;
  if(!s_model) {
    cout << "Haven't trained svm when calling, exiting" << endl;
    exit(-1);
  }
  struct svm_node *node = Malloc(struct svm_node, dim+1);
  for(int i = 0; i < dim; i++) {
    node[i].index = i;
    node[i].value = x[i];
  }
  node[dim+1].index = -1;
  double result = svm_predict(s_model, node);
  free(node);
  return result;
}
