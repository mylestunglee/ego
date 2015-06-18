#include "ego.h"
#include "surrogate.h"
#include "functions.h"
#include "optimise.h"
#include <vector>

using namespace std;

int main(int argc, char * argv[]) {
  string python_name = "/homes/wjn11/MLO/examples/quadrature_method_based_app";
  vector<double> lower = {1.0, 11.0, 4.0};
  vector<double> upper = {16.0, 53.0, 32.0};
  vector<double> gamma, C;
  int dimension = 3;
  double max_f = 0.0390423230495;

  Surrogate *sg = new Surrogate(dimension, SEard, true, true);

  EGO *ego = new EGO(dimension, sg, lower, upper, python_name, 1);
  ego->suppress = true;
  for(int i = lower[0]; i <= upper[0]; i++) {
    for(int j = lower[1]; j <= upper[1]; j++) {
      for(int k = lower[2]; k <= upper[2]; k++) {
        vector<double> x(dimension, 0.0);
	x[0] = i;
	x[1] = j;
	x[2] = k;
	ego->python_eval(x);
	ego->update_running();
      }
    }
  }
  delete ego;


  //double (*fitness)(double x[]) = &sphere_5;
  //int dimension = 2;
  //  
  //vector<double> lower;
  //vector<double> upper;
  //for(int i = 0; i< dimension; i++) {
  //  lower.push_back(-5.0);
  //  upper.push_back(5.0);
  //}

  ////cout << "Building" << endl;
  ////for(int i = 3; i < 8; i++) {
  //  //dimension = i;
  //  Surrogate *sg = new Surrogate(dimension, SEiso);

  //  EGO *ego = new EGO(dimension, sg, lower, upper, fitness);

  //  ego->search_type = 1;
  //  //ego->max_fitness = -39 * dimension;
  //  ego->max_fitness = 0;
  //  ego->suppress = true;
  //  ego->is_discrete = true;
  //  ego->n_sims = 50;
  //  //ego->max_points = 1000;
  //  ego->max_points = upper[0] - lower[0];
  //  ego->num_points = ego->max_points;
  //  ego->pso_gen = 100;
  //  ego->exhaustive = true;


  //  //cout << "Sample"<<endl;
  //  //ego->sample_plan(10*dimension, 5);
  //  ego->latin_hypercube(5,5);
  //  ego->sg->train();
  //  //cout << "Sampled"<<endl;
  //  ego->suppress = true;
  //  opt *op = new opt(2, upper, lower, ego, false);
  //  vector<vector<double>> best = op->combined_optimise(vector<double>(2,3.0), 3, 6, 2);
  //  cout << endl << endl << endl << endl;
  //  for(int k = 0; k < best.size(); k++) {
  //    for(int j = 0; j < dimension; j++) {
  //      cout << best[k][j] << ", ";
  //    }
  //    cout << endl;
  //  }


  //  //for(int j = 1; j < 4; j++) {
  //  //  ego->search_type = j;
  //  //  //cout <<"Run dimension "<<dimension << endl;
  //  //  auto t1 = std::chrono::high_resolution_clock::now();
  //  //  ego->max_ei_par(1);
  //  //  auto t2 = std::chrono::high_resolution_clock::now();
  //  //  auto t3 = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
  //  //  cout << "Dimension " << dimension << " search " << j << " took " << t3  << " iter=" << ego->iter << " / " << ego->num_iterations<< endl;
  //  //  //cout << "In FPGA time took " << ego->total_time << endl;
  //  //}
  //  delete ego;
  ////}
}
