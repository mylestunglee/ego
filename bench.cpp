#include "ego.h"
#include "surrogate.h"
#include "functions.h"
#include <vector>
#include <chrono>

using namespace std;


shared_ptr<EGO> reset_ego()
{
  //double (*fitness)(double x[]) =  &sphere_3;
  //int dimension = 3;
  //vector<double> lower = {-5.0, -5.0, -5.0};
  //vector<double> upper = {5.0, 5.0, 5.0};

  shared_ptr<EGO> ego;
  double (*fitness)(double x[]) =  &sphere_3;
  int dimension = 3;
  vector<double> lower = {-13.0, -13.0, -13.0};
  vector<double> upper = {13.0, 13.0, 13.0};

  //double (*fitness)(double x[]) =  &sphere_4;
  //int dimension = 4;
  //vector<double> lower = {-5.0, -5.0, -5.0, -5.0};
  //vector<double> upper = {5.0, 5.0, 5.0, 5.0};
  
  //cout << "Building" << endl;
  shared_ptr<Surrogate> sg(new Surrogate(dimension, SEiso));
  sg->set_params(0.0, 0.0);
  ego = make_shared<EGO>(dimension, sg, lower, upper, fitness);
  //cout << "Built" << endl;

  ego->suppress = true;
  ego->is_discrete = true;
  ego->n_sims = 10;
  ego->max_points = upper[0] - lower[0];
  ego->num_points = ego->max_points;
  ego->pso_gen = 100;
  //cout << "sampling" << endl;
  //ego->sample_plan(10, 5);

  ego->sample_plan(ego->max_points, 5);
  //cout << "finished resetting" << endl;
  return ego;
}

int main(int argc, char * argv[]) {

  shared_ptr<EGO> ego = NULL;

  //for(int i = 1; i < 3; i++) {
  //  auto t1 = std::chrono::high_resolution_clock::now();
  //  ego->brute_search_swarm(10, i);
  //  auto t2 = std::chrono::high_resolution_clock::now();
  //  auto t3 = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
  //  cout << "Brute search npts=" << npts[i] << " l=2, took " << t3 << endl;
  //  ego = reset_ego();
  //}

  //for(int i = 2; i < 7; i++) {
  //  ego = reset_ego();
  //  ego->n_sims = i * 10 / 2;
  //  ego->n_sims = 10;
  //  ego->use_brute_search = false;
  //  ego->swarm = false;
  //  ego->pso_gen = 500;
  //  auto t1 = std::chrono::high_resolution_clock::now();
  //  vector<double> x = ego->max_ei_par(i);
  //  auto t2 = std::chrono::high_resolution_clock::now();
  //  auto t3 = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
  //  cout << "PSO search l=" << i << " nsims " << ego->n_sims << ", took " << t3 << endl;
  //  cout << "Fitness = " << ego->fitness(x) << endl;
  //  ego->n_sims = 1000;
  //  cout << "Fitness = " << ego->fitness(x) << endl;
  //}

  //for(int i = 2; i < 7; i++) {
  //  ego = reset_ego();
  //  ego->num_points = ego->max_points;
  //  ego->n_sims = 10;
  //  //ego->n_sims = i * 10 / 2;
  //  ego->use_brute_search = true;
  //  ego->swarm = true;
  //  auto t1 = std::chrono::high_resolution_clock::now();
  //  vector<double> x = ego->max_ei_par(i);
  //  auto t2 = std::chrono::high_resolution_clock::now();
  //  auto t3 = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
  //  cout << "Brute swarm search l=" << i << " nsims " << ego->n_sims << ", took " << t3 << endl;
  //  cout << "Fitness = " << ego->fitness(x) << endl;
  //  ego->n_sims = 1000;
  //  cout << "Fitness = " << ego->fitness(x) << endl;
  //}



  for(int i = 2; i < 6; i++) {
    for(int j = 0; j < 4; j++) {
    ego = reset_ego();
    ego->num_lambda = i;
    ego->use_brute_search = true;
    ego->swarm = true;
    auto t1 = std::chrono::high_resolution_clock::now();
    ego->run();
    auto t2 = std::chrono::high_resolution_clock::now();
    auto t3 = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
    cout << "Brute Swarm search l=" << i << " took " << t3  << " iter=" << ego->iter << " / " << ego->num_iterations<< endl;
    }
  }
  

  for(int i = 3; i < 6; i++) {
    for(int j = 0; j < 4; j++) {
    ego = reset_ego();
    ego->num_lambda = i;
    ego->use_brute_search = false;
    ego->pso_gen = 20;
    auto t1 = std::chrono::high_resolution_clock::now();
    ego->run();
    auto t2 = std::chrono::high_resolution_clock::now();
    auto t3 = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
    cout << "PSO search l=" << i << " took " << t3  << " iter=" << ego->iter << " / " << ego->num_iterations<< endl;
    }
  }
}
