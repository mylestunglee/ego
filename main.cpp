#include "ego.h"
#include "surrogate.h"
#include "functions.h"
#include <vector>

using namespace std;

int main(int argc, char * argv[]) {

  vector<double> lower(30, -5.0);
  vector<double> upper(30, 5.0);
  //for(int i = 0; i< 30; i++) {
  //  lower.push_back(-5.0);
  //  upper.push_back(5.0);
  //}
  vector<double(*)(double x[])> fitness_functions = {&sphere_5, &sphere_10,&sphere_15, &sphere_20, &sphere_25, &sphere_30};

  for(size_t i = 0; i < fitness_functions.size(); i++) {
    cout << "Building" << endl;
    int dimension = (i+1)*3;
    double (*fitness)(double x[]) = fitness_functions[i];
    
    Surrogate *sg = new Surrogate(dimension, SEard);

    EGO *ego = new EGO(dimension, sg, lower, upper, fitness);

    ego->search_type = 1;
    //ego->max_fitness = -39 * dimension;
    ego->suppress = true;
    ego->is_discrete = true;
    ego->n_sims = 50;
    //ego->max_points = 1000;
    ego->max_points = upper[0] - lower[0];
    ego->num_points = ego->max_points;
    ego->pso_gen = 100;


    //cout << "Sample"<<endl;
    //ego->sample_plan(10*dimension, 5);
    ego->latin_hypercube(10,5);
    ego->sg->train();
    //cout << "Sampled"<<endl;
    ego->suppress = false;


    cout <<"Run" << endl;
    auto t1 = std::chrono::high_resolution_clock::now();
    ego->max_ei_par(1);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto t3 = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
    cout << "Brute search tang dim = " << dimension << " took " << t3  << " iter=" << ego->iter << " / " << ego->num_iterations<< endl;
    //cout << "In FPGA time took " << ego->total_time << endl;
    delete ego;
  }
}
