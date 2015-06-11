#include "ego.h"
#include "surrogate.h"
#include "functions.h"
#include <vector>
#include <chrono>

using namespace std;
EGO *reset_ego();

int dimension;


EGO *reset_ego()
{
  double max_f = 0;
  //double (*fitness)(double x[]) =  &sphere_3;
  //int dimension = 3;
  //vector<double> lower = {-13.0, -13.0, -13.0};
  //vector<double> upper = {13.0, 13.0, 13.0};

  //double (*fitness)(double x[]) =  &sphere_20;
  //vector<double> lower, upper;
  //for(int i = 0; i < 25; i++) {
  //  lower.push_back(-5.0);
  //  upper.push_back(5.0);
  //}

  double (*fitness)(double x[]) =  &prob09;
  vector<double> lower, upper;
  for(int i = 0; i < 2; i++) {
    lower.push_back(-2.0);
    upper.push_back(2.0);
  }
  max_f = 0.001;

  //double (*fitness)(double x[]) =  &easy_test2;
  //int dimension = 1;
  //vector<double> lower = {-10.0, -13.0, -13.0};
  //vector<double> upper = {10.0, 13.0, 13.0};

  //cout << "Building" << endl;
  Surrogate *sg = new Surrogate(dimension, SEiso, true);
  //sg->set_params(0.0, 0.0);
  EGO *ego = new EGO(dimension, sg, lower, upper, fitness);
  //cout << "Built" << endl;

  ego->max_fitness = max_f;
  ego->suppress = true;
  ego->is_discrete = false;
  ego->n_sims = 10;
  ego->max_points = upper[0] - lower[0];
  ego->max_points = 100;
  ego->num_points = ego->max_points;
  ego->pso_gen = 100;

  //cout << "Sample"<<endl;
  ego->sample_plan(20, 5);
  //cout << "Sampled"<<endl;
  return ego;
}

int main(int argc, char * argv[]) 
{
  dimension = 8;
  EGO* ego = NULL;
  if(argc > 1) {
    dimension = atoi(argv[1]);
  }

  //ego = reset_ego();
  //ego->suppress = false;
  //ego->use_brute_search = true;
  //vector<double> x = {6};
  //ego->evaluate(x);
  //x = {4};
  //ego->evaluate(x);
  //x = {8};
  //ego->evaluate(x);
  //x = {-5};
  //ego->evaluate(x);
  //ego->check_running_tasks();
  //ego->sg->train();
  ////ego->run();
  ////ego->sg->train();
  //for(int i = 0; i < 10; i++) {
  //x[0] = i;
  //pair<double, double> p = ego->sg->predict(&x[0]);
  //cout << i << " " << ego->ei(p.first, p.second, ego->best_fitness) << " " << p.first << " " << p.second << " " << ego->sg->svm_label(&x[0]) << endl;
  //}

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

  //for(int i = 1; i < 6; i++) {
  //  for(int j = 0; j < 2; j++) {
  //  ego = reset_ego();
  //  ego->use_brute_search = true;
  //  auto t1 = std::chrono::high_resolution_clock::now();
  //  ego->brute_search_swarm(10, i);
  //  auto t2 = std::chrono::high_resolution_clock::now();
  //  auto t3 = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
  //  cout << "Brute swarm loop w/ calc dim=" << ego->dimension << " l=" << i << ", took " << t3 << endl;
  //  delete ego;
  //  }
  //}


  for(int i = 1; i < 6; i++) {
    //ego = reset_ego();
    //ego->num_lambda = 3;
    //ego->use_brute_search = true;
    //ego->suppress = false;
    //auto t1 = std::chrono::high_resolution_clock::now();
    //ego->max_ei_par(i);
    //auto t2 = std::chrono::high_resolution_clock::now();
    //auto t3 = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
    //cout << "Brute search l=" << i << " took " << t3  << " iter=" << ego->iter << " / " << ego->num_iterations<< endl;
    //delete ego;

 
    for(int j = 5; j < 50000; j*=10) {
    for(int k = 0; k <= 5; k++) {
    //for(int k = 100; k <= 700; k += 100) {
      ego = reset_ego();
      //ego->sample_plan(ego->dimension, 5);
      ego->num_lambda = i;
      ego->n_sims = j;
      ego->use_brute_search = false;
      ego->population_size = 500;
      ego->suppress = true;
      auto t1 = std::chrono::high_resolution_clock::now();
      ego->run();
      auto t2 = std::chrono::high_resolution_clock::now();
      auto t3 = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
      cout << "PSO search fit="<< ego->best_fitness << ego->dimension << " l=" << i << " pop=" << 500 << " nsims=" << j << " iter=" << ego->iter << "/" << ego->num_iterations << " took " << t3 << endl;
      delete ego;
    }
    }
  }

  //for(int i = 2; i < 6; i++) {
  //  for(int j = 0; j < 4; j++) {
  //  ego = reset_ego();
  //  ego->num_lambda = i;
  //  ego->use_brute_search = false;
  //  ego->pso_gen = 20;
  //  ego->suppress = false;
  //  auto t1 = std::chrono::high_resolution_clock::now();
  //  ego->run();
  //  auto t2 = std::chrono::high_resolution_clock::now();
  //  auto t3 = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
  //  cout << "PSO search l=" << i << " took " << t3  << " iter=" << ego->iter << " / " << ego->num_iterations<< endl;
  //  delete ego;
  //  }
  //}
}
