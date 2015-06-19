#include "ego.h"
#include "surrogate.h"
#include "functions.h"
#include <vector>
#include <chrono>

using namespace std;
EGO *reset_ego();

int dimension;
example bench;
int search_type;
int lambda;
int n_sims;
bool use_cost;
bool use_log;

EGO *reset_ego()
{
  long double max_f = 0;
  vector<double> lower;
  vector<double> upper;
  vector<double> gamma, C;
  string python_name;
  vector<double> always_valid;
  bool exhaustive = true;
  switch(search_type) {
    case 1:
      cout <<"Using Brute Search" << endl;
      break;

    case 2:
      cout <<"Using PS0 Search" << endl;
      break;

    case 3:
      cout <<"Using combined PSO and Brute Search" << endl;
      break;

    default:
      cout << "Didn't set search type, defaulting to PSO" << endl;
      search_type = 2;
      break;
  }

  switch(bench) {
    case quad:
      python_name = "/homes/wjn11/MLO/examples/quadrature_method_based_app";
      lower = {1.0, 11.0, 4.0};
      upper = {16.0, 53.0, 32.0};
      dimension = 3;
      //max_f = 0.0390423230495;
      max_f = 0.05785136;

      //if(local) {
      //  for(double i = -10.; i < 31; i++) {
      //    gamma.push_back(10 * pow(1.25, i));
      //  }
      //  for(double i = -30.; i < 31; i++) {
      //    C.push_back(pow(1.5, i));
      //  }
      //} else {
      for(double i = -20.; i < 21; i++) {
        gamma.push_back(pow(1.2, i));
        C.push_back(10 * pow(1.25, i));
      }
      //}
      always_valid = {1., 53., 32.};
      if(search_type == 1) {
        exhaustive = true;
      }
      break;

    case pq:
      python_name = "/homes/wjn11/MLO/examples/pq";
      lower = {4.0, 80.0, 1.0};
      upper = {53.0, 120.0, 4.0};
      dimension = 3;
      max_f = 153.0;
      exhaustive = true;
      for(double i = -10.; i < 11; i++) {
        gamma.push_back(pow(1.2, i));
      }
      for(double i = 1.; i < 11; i++) {
        C.push_back(10 * pow(1.25, i));
      }
      always_valid = {4., 80., 1.};
      break;

    case rtm:
      python_name = "/homes/wjn11/MLO/examples/xinyu_rtm";
      lower = {1.0, 1.0, 4.0, 1.0, 1.0, 1.0, 1.0};
      upper = {10.0, 10.0, 24.0, 3.0, 10.0, 10.0, 32.0};
      always_valid = {1.0, 1.0, 4.0, 1.0, 1.0, 1.0, 1.0};
      dimension = 7;
      max_f = 0.07;
      break;
  }

  cout << "Building" << endl;
  Surrogate *sg = new Surrogate(dimension, SEard, true, use_log);
  sg->gamma = gamma;
  cout << sg->gamma.size() << " gamma size" << endl;
  sg->C = C;

  EGO *ego = new EGO(dimension, sg, lower, upper, python_name, search_type);
  cout << "Built" << endl;

  ego->search_type = search_type;
  ego->max_fitness = max_f;
  ego->max_iterations = 1000;
  ego->suppress = false;
  ego->is_discrete = true;
  ego->n_sims = n_sims;
  ego->use_cost = use_cost;
  ego->num_lambda = lambda;
  ego->max_points = upper[1] - lower[1];
  if(bench == pq) {
    ego->is_max = true;
    ego->best_fitness = -100000;
  }
  //cout << "MAX = " << ego->is_max << endl;
  //ego->max_points = 100;
  ego->num_points = ego->max_points;
  ego->pso_gen = 500;
  ego->population_size = 200;
  //ego->use_brute_search = use_brute;
  ego->exhaustive = exhaustive;

  ego->python_eval(always_valid);
  cout << "Sample"<<endl;
  ego->sample_plan(10*dimension, 10);
  cout << "Sampled"<<endl;
  return ego;
}

int main(int argc, char * argv[]) 
{
  srand(time(NULL));
  dimension = 3;
  EGO* ego = NULL;
  if(argc < 6) {
    cout<<"Usage: " << endl;
    cout <<"./test example search_type use_cost lambda n_sims" << endl;
    cout << "examples: 1 = Quad, 2 = PQ, 3 = RTM" << endl;
    cout << "search_type: 1 = Brute, 2 = PSO, 3 = PSO + Brute" << endl;
    cout << "use_cost: 0 = Standard EI, 1 = EI / Cost" << endl;
    exit(0);
  } else {
    bench = static_cast<example>(atoi(argv[1]));
    search_type = atoi(argv[2]);
    use_cost = atoi(argv[3]);
    lambda = atoi(argv[4]);
    n_sims = atoi(argv[5]);
    use_log = atoi(argv[6]);
    cout << bench <<" "<<search_type<<use_cost<<lambda<<n_sims<<endl;
  }

  //vector<double> lambdas = {1, 2, 4, 6};
  //for(size_t i = 0; i < lambdas.size(); i++) {
      //cout << endl << endl << endl << endl << endl;
      //bench = quad;
      //search_type = 1;
      //use_cost = 1;
      //lambda = lambdas[i];
      //n_sims = 100;
      //use_log = 1;
      //cout << "STARTING BENCH OF QUAD ";
      //cout << "LAMBDA = " << lambda;
      //cout << " SEARCH = BRUTE ";
      //cout << " NSIMS = " << n_sims;
      //cout << " COST";
      ego = reset_ego();
      ego->suppress = false;
      auto t1 = std::chrono::high_resolution_clock::now();
      ego->run_quad();
      auto t2 = std::chrono::high_resolution_clock::now();
      auto t3 = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
      cout << "Search l=" << lambda << " took " << t3  << " iter=" << ego->iter << " / " << ego->num_iterations<< endl;
      cout << "In FPGA time took " << ego->total_time << endl;
      delete ego;
  //}
}
