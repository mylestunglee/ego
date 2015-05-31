#include "ego.h"
#include "surrogate.h"
#include "functions.h"
#include <vector>
#include <chrono>

using namespace std;

EGO *reset_ego(EGO *ego=NULL)
{
  if(ego) {
    delete ego->sg;
    delete ego;
  }
  //double (*fitness)(double x[]) =  &sphere_3;
  //int dimension = 3;
  //vector<double> lower = {-5.0, -5.0, -5.0};
  //vector<double> upper = {5.0, 5.0, 5.0};
  double (*fitness)(double x[]) =  &sphere_4;
  int dimension = 4;
  vector<double> lower = {-5.0, -5.0, -5.0, -5.0};
  vector<double> upper = {5.0, 5.0, 5.0, 5.0};
  
  Surrogate *sg = new Surrogate(dimension, SEiso);
  sg->set_params(0.0, 0.0);
  ego = new EGO(dimension, sg, lower, upper, fitness);

  ego->suppress = true;
  ego->max_fitness = 0;
  ego->max_iterations = 1000;
  ego->min_expected_imp = 10;
  ego->is_discrete = true;
  ego->n_sims = 50;
  ego->population_size = 5;
  ego->num_lambda = 5;
  ego->num_points = 5;
  ego->use_brute_search = true;
  ego->sample_plan(10, 5);
  return ego;
}

int main(int argc, char * argv[]) {

  EGO *ego = NULL;

  ego = reset_ego(ego);
  ego->num_points = 10;
  ego->use_brute_search = true;
  ego->swarm = true;
  ego->n_sims = 5;
  ego->population_size = 100;

  int npts[] = {5, 10};
  ////for(int i = 0; i < 2; i++) {
  ////  auto t1 = std::chrono::high_resolution_clock::now();
  ////  ego->brute_search_loop(npts[i], 2);
  ////  auto t2 = std::chrono::high_resolution_clock::now();
  ////  auto t3 = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
  ////  cout << "Brute search npts=" << npts[i] << " l=2, took " << t3 << endl;
  ////}
  //for(int i = 0; i < 2; i++) {
  //  auto t1 = std::chrono::high_resolution_clock::now();
  //  vector<double> x = ego->max_ei_par(4);
  //  auto t2 = std::chrono::high_resolution_clock::now();
  //  auto t3 = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
  //  cout << "PSO search l=4, took " << t3 << endl;
  //  ego->n_sims = 1000;
  //  cout << "Fitness = " << ego->fitness(x) << endl;
  //}

  //ego->n_sims = 5;
  //for(int i = 0; i < 2; i++) {
  //  auto t1 = std::chrono::high_resolution_clock::now();
  //  vector<double> *best = ego->brute_search_swarm(10, 4);
  //  auto t2 = std::chrono::high_resolution_clock::now();
  //  auto t3 = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
  //  cout << "Brute swarm search l=4, took " << t3 << endl;
  //  ego->n_sims = 1000;
  //  if(best) {
  //    cout << "Fitness = " << ego->fitness(*best) << endl;
  //  }
  //}



  for(int i = 3; i < 5; i++) {
    for(int j = 0; j < 3; j++) {
    ego = reset_ego(ego);
    ego->num_lambda = i;
    ego->n_sims = 10;
    ego->num_points = 10;
    ego->use_brute_search = true;
    ego->swarm = true;
    ego->suppress = true;
    auto t1 = std::chrono::high_resolution_clock::now();
    ego->run();
    auto t2 = std::chrono::high_resolution_clock::now();
    auto t3 = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
    cout << "Brute Swarm search l=" << i << " took " << t3 << endl;
    }
  }
  //for(int i = 1; i < 3; i++) {
  //  for(int j = 0; j < 3; j++) {
  //  ego = reset_ego(ego);
  //  ego->num_lambda = i;
  //  ego->n_sims = 5;
  //  ego->num_points = 5;
  //  ego->use_brute_search = true;
  //  auto t1 = std::chrono::high_resolution_clock::now();
  //  ego->run();
  //  auto t2 = std::chrono::high_resolution_clock::now();
  //  auto t3 = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
  //  cout << "Brute search l=" << i << " took " << t3 << endl;
  //  }
  //}
  for(int i = 3; i < 5; i++) {
    for(int j = 0; j < 3; j++) {
    ego = reset_ego(ego);
    ego->num_lambda = i;
    ego->n_sims = 10;
    ego->num_points = 10;
    ego->use_brute_search = false;
    auto t1 = std::chrono::high_resolution_clock::now();
    ego->run();
    auto t2 = std::chrono::high_resolution_clock::now();
    auto t3 = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
    cout << "PSO search l=" << i << " took " << t3 << endl;
    }
  }
}
