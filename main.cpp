#include "ego.h"
#include "surrogate.h"
#include "functions.h"
#include <vector>

using namespace std;

int main(int argc, char * argv[]) {

  double (*fitness)(double x[]) = &sphere_3;
  int dimension = 3;
    
  vector<double> lower;
  vector<double> upper;
  for(int i = 0; i< dimension; i++) {
    lower.push_back(-5.0);
    upper.push_back(5.0);
  }

  cout << "Building" << endl;
  for(int i = 1; i < 5; i++) {
    Surrogate *sg = new Surrogate(dimension, SEiso);

    EGO *ego = new EGO(dimension, sg, lower, upper, fitness);

    ego->search_type = 1;
    //ego->max_fitness = -39 * dimension;
    ego->max_fitness = 0;
    ego->suppress = false;
    ego->is_discrete = true;
    ego->n_sims = 50;
    ego->num_lambda = i;
    ego->max_points = 1000;
    ego->num_points = ego->max_points;
    ego->pso_gen = 100;


    cout << "Sample"<<endl;
    //ego->sample_plan(10*dimension, 5);
    ego->latin_hypercube(10*dimension,5);
    cout << "Sampled"<<endl;
    ego->num_lambda = i;
    ego->suppress = false;


    auto t1 = std::chrono::high_resolution_clock::now();
    ego->run();
    auto t2 = std::chrono::high_resolution_clock::now();
    auto t3 = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
    cout << "Optimise Sphere l=" << i << " took " << t3  << " iter=" << ego->iter << " / " << ego->num_iterations<< endl;
    //cout << "In FPGA time took " << ego->total_time << endl;
    delete ego;
  }
}
