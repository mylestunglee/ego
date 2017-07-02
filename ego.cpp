#include "ego.hpp"
#include "optimise.hpp"
#include "ihs.hpp"
#include <ctime>
#include <thread>
#include <chrono>

#include <gsl/gsl_cdf.h>
#include <gsl/gsl_randist.h>

using namespace std;

EGO::EGO(vector<pair<double, double>> boundaries, Evaluator& evaluator) :
	evaluator(evaluator)
{
  dimension = boundaries.size();

	for (auto boundary : boundaries) {
		lower.push_back(boundary.first);
		upper.push_back(boundary.second);
	}

	sg = new Surrogate(boundaries.size(), SEiso, true, false);
	sg_cost = new Surrogate(boundaries.size(), SEard);

	rng = gsl_rng_alloc(gsl_rng_taus);

  n_sims = 50;
  max_iterations = 100;
  num_iterations = 0;
  num_lambda = 3;
  lambda = num_lambda;
  population_size = 100;
  num_points = 10;
  max_points = 10;
  pso_gen = 1;
  best_fitness = 10000000000;
  is_discrete = false;
  use_brute_search = false;
  suppress = false;
  search_type = 2;
}

EGO::~EGO() {
	gsl_rng_free(rng);

	delete sg;
	delete sg_cost;
}

void EGO::run_quad()
{
  sg->train_gp_first();
  while(num_iterations < max_iterations) {
    std::clock_t t3 = std::clock();
    sg->train();
    sg_cost->train();

    t3 = (std::clock() - t3) / CLOCKS_PER_SEC;
    update_running(t3);

	cout << "Iteration: " << num_iterations << endl;

    if(lambda > 0) {
      int temp_lambda = lambda;
      //t1 = std::chrono::high_resolution_clock::now();
      t3 = std::clock();
      vector<double> best_xs = max_ei_par(temp_lambda);
      //t2 = std::chrono::high_resolution_clock::now();
      //t3 = std::chrono::duration_cast<std::chrono::seconds>(t2-t1).count();
      t3 = (std::clock() - t3) / CLOCKS_PER_SEC;
      update_running(t3);

      t3 = std::clock();
      for(int l = 0; l < temp_lambda; l++) {
        vector<double> y(dimension, 0.0);
        for(int j = 0; j < dimension; j++) {
          y[j] = best_xs[l * dimension + j];
        }

        if(not_running(&y[0]) && not_run(&y[0])) {
          python_eval(y);
        } else {
	  if(!suppress) {
            cout << "Have run: ";
            for(int j = 0; j < dimension; j++) {
              cout << y[j] << " ";
            }
	  }
          y = local_random();
	  int s = y.size();
	  if(s != dimension) {
	    break;
	  }
          python_eval(y);
        }
      }
      t3 = (std::clock() - t3) / CLOCKS_PER_SEC;
      update_running(t3);
    }
    if(lambda == 0) update_running();
    if(running.size() >= num_lambda) {
      cout << "Couldn't update running, exiting" << endl;
      exit(-1);
    }
  }
}

void EGO::python_eval(vector<double> &x, bool add) {
	if(!add) {
		pair<double, double> p = sg->predict(&x[0], true);
		mu_means.push_back(p.first);
		mu_vars.push_back(p.second);
	}

  struct running_node run;
  vector<double> result = evaluator.evaluate(x);
	run.fitness = result[0];
	run.label = result[1];
	run.cost = result[2];
  run.addReturn = 0;

  if(add) {
    //We evaluated it, now add to all our surrogate models
    num_iterations++;

    int label = 2 - (int) (run.label == 0); //1 if run.label == 0
    sg->add(x, run.fitness, label, run.addReturn);
    if(run.addReturn == 0) {
        sg_cost->add(x, run.cost);
    }
    if(run.label == 0 ) {
      valid_set.push_back(x);
        if(run.fitness < best_fitness) {
          best_fitness = run.fitness;
          best_particle = x;
        }
        //for(int i = 0; i < dimension; i++) cout << x[i] << " ";
        //cout << run.fitness << endl;
      training_f.push_back(run.fitness);
    }
    training.push_back(x);
  } else {
    //About to run, stick on running vector
    run.is_finished = false;
    run.data = x;
    run.pos = mu_means.size() - 1;
    running.push_back(run);
  }
}

void EGO::update_running(const long int &t)
{
  long int time = t;
  if(time == -1L) {
    time = 1000000000000000L;
    for(vector<struct running_node>::iterator node = running.begin(); node != running.end(); node++) {
      time = min(time, (long) node->cost);
    }
    time++;
  }
  if(time > 0) {
    for(vector<struct running_node>::iterator node = running.begin(); node != running.end();) {
      node->cost -= time;
      if(node->cost <= 0) {
        python_eval(node->data, true);

        //Delete estimations
        mu_means.erase(mu_means.begin() + node->pos);
        mu_vars.erase(mu_vars.begin() + node->pos);
        for(size_t i = 0; i < running.size(); i++) {
          if(running[i].pos > node->pos) running[i].pos--;
        }

        //Delete node from running vector
        node = running.erase(node);
      } else {
        node++;
      }
    }
  }
  lambda = num_lambda - running.size();
}



void EGO::run()
{
  suppress = false;
  if(!suppress) cout << "Started, dim=" << dimension << ", lambda=" << num_lambda << endl;
  sg->train_gp_first();
  while(num_iterations < max_iterations) {
    //Check to see if any workers have finished computing
    std::clock_t t3 = std::clock();
    check_running_tasks();
    sg->train();
    //t3 = (std::clock() - t3) / CLOCKS_PER_SEC;


    int lambda = num_lambda - running.size();
    lambda = min(lambda, max_iterations - num_iterations);

	cout << "Iterations: " << num_iterations << endl;

    //t3 = std::clock();
    if(lambda > 0) {
      cout << " max " <<endl;
      vector<double> best_xs = max_ei_par(lambda);
      cout << " max " <<endl;
      for(int l = 0; l < lambda; l++) {
        vector<double> y(dimension, 0.0);
        for(int j = 0; j < dimension; j++) {
          y[j] = best_xs[l * dimension + j];
        }

        if(not_running(&y[0]) && not_run(&y[0])) {
          evaluate(y);
        } else {
	  if(!suppress) {
            cout << "Have run: ";
            for(int j = 0; j < dimension; j++) {
              cout << y[j] << " ";
            }
	  }
          y = brute_search_local_swarm(best_particle, 1, 1, true);
          evaluate(y);
        }
      }
    }
    t3 = (std::clock() - t3) / CLOCKS_PER_SEC;
    cout << "Lambda" << lambda << endl;
  }
  check_running_tasks();
}

vector<double> EGO::best_result()
{
  return best_particle;
}

void EGO::evaluate(const vector<double> &x)
{
  double *data = (double *) &x[0];

  double mean, var;
  if(sg->is_trained){
    pair<double, double> p = sg->predict(data, true);
    mean = p.first;
    var = p.second;
  } else {
    mean = sg->mean(data);
    var = sg->var(data);
  }
  mu_means.push_back(mean);
  mu_vars.push_back(var);

  //Add to running set
  struct running_node run;
  run.fitness = mean;
  run.is_finished = false;
  run.data = x;
  run.pos = mu_means.size() - 1;
  running.push_back(run);
  num_iterations++;

  //Launch another thread to calculate
//  thread (&EGO::worker_task, this, run).detach();

  worker_task(run);
}

/* Thread function that manages evaluation storage */
void EGO::worker_task(struct running_node& run) {
	double fitness = evaluator.evaluate(run.data)[0];

	running_mtx.lock();

	run.is_finished = true;
	run.fitness = fitness;
	run.label = 1;

	running_mtx.unlock();
}

void EGO::check_running_tasks() {
	running_mtx.lock();

	vector<struct running_node>::iterator node = running.begin();

	while(node != running.end()) {
		if(node->is_finished) {
			//Add it to our training set
			training.push_back(node->data);

			if(sg->is_svm) {
				sg->add(node->data, node->fitness, node->label);
			} else {
				sg->add(node->data, node->fitness);
			}

			update_best_result(node->data, node->fitness);

			//Delete estimations
			mu_means.erase(mu_means.begin() + node->pos);
			mu_vars.erase(mu_vars.begin() + node->pos);

			for (auto& run : running) {
				if (run.pos > node->pos) {
					run.pos--;
				}
			}

			//Delete node from running vector
			node = running.erase(node);
		} else {
			node++;
		}
	}

	running_mtx.unlock();
}

double EGO::fitness(const vector<double> &x)
{
  int num = x.size() / dimension;

  double lambda_means[num];
  double lambda_vars[num];
  double cost = 1.0, mean_cost=1.0;

  for(int i = 0; i < num; i++) {
    double y [dimension];

    for(int j = 0; j < dimension; j++) {
      y[j] = x[i * dimension + j];
    }

    if(not_running(y)) {
      pair<double, double> p = sg->predict(y, true);
      lambda_means[i] = p.first;
      lambda_vars[i] = p.second;

        pair<double, double> p_cost = sg_cost->predict(y);
	cost += p_cost.first;
	mean_cost = sg_cost->mean_fit;

    } else {
      lambda_means[i] = 100000000000;
      lambda_vars[i] = 0;
    }
  }

  double result = ei_multi(lambda_vars, lambda_means, num, n_sims, sg->best_raw());
  if(cost > 0) {
    result = result * mean_cost / cost;
  }
  result = exp(abs(result/n_sims) - 1) * 100;
  return result;
}

vector<double> EGO::max_ei_par(int llambda)
{
  vector<double> best;
  static bool use_mean = false;
  if(search_type == 1) {
    vector<double> *ptr = brute_search_swarm(num_points, llambda, use_mean);
    if(ptr) {
      best = *ptr;
      delete ptr;
    } else {
      if(!suppress) cout << "Couldn't find new particles, searching in region of best" << endl;
      best = local_random(1.0, llambda);
      if(!suppress) {
        for(size_t i = 0; i < best.size(); i++) {
          cout << best[i] << " ";
        }
        cout << " got around best" << endl;
      }
    }
  } else if(search_type == 2){
    int size = dimension * llambda;
    vector<double> low(size, 0.0), up(size, 0.0), x(size, 0.0);

    for(int i = 0; i < size; i++) {
      low[i] = lower[i % dimension];
      up[i] = upper[i % dimension];
      x[i] = best_particle[i % dimension];
    }

    opt *op = new opt(size, up, low, this, is_discrete);
    best = op->swarm_optimise(x, pso_gen, population_size * size);
    double best_f = op->best_part->best_fitness;

    if(best_f < 0.001) {
      best = local_random(1.0, llambda);
    }
    if(!suppress) {
      cout << "[";
      for(int i = 0; i < llambda; i++) {
        for(int j = 0; j < dimension; j++) {
          cout << best[i*dimension + j] << " ";
        }
        cout << "\b; ";
      }
      cout << "\b\b] = best = "  << best_f << endl;
    }
    delete op;
   } else {
    int size = dimension * llambda;
    vector<double> low(size, 0.0), up(size, 0.0), x(size, 0.0);

    for(int i = 0; i < size; i++) {
      low[i] = lower[i % dimension];
      up[i] = upper[i % dimension];
      x[i] = best_particle[i % dimension];
    }

    opt *op = new opt(size, up, low, this, is_discrete);
    vector<vector<double>> swarms = op->combined_optimise(x, pso_gen, population_size * size, lambda);
    double best_f = 0.0;
    if(swarms.size() <= (unsigned) llambda) {
      best = local_random(1.0, llambda);
    } else {
      for(size_t i = 0; i < swarms[llambda].size(); i++) {
        best_f = max(best_f, swarms[lambda][i]);
      }
      if(best_f < 0.0001) {
        best = local_random(1.0, llambda);
        best_f = fitness(best);
      } else {
        int old_n_sims = n_sims;
        n_sims *= 10;
	vector<double> x, y;
        for(int i = 0; i < llambda; i++) {
	  x = swarms[i];
          y = brute_search_local_swarm(x, 1, 1, true, false, use_mean);
	  if(y.size() < (unsigned) size) {
	    y = local_random();
	  }
          for(int j = 0; j < dimension; j++) {
            best.push_back(y[j]);
          }
        }
	x.clear();
	y.clear();
        best_f = fitness(best);
        n_sims = old_n_sims;
      }
    }
    if(!suppress) {
      cout << "[";
      for(int i = 0; i < llambda; i++) {
        for(int j = 0; j < dimension; j++) {
          cout << best[i*dimension + j] << " ";
        }
        cout << "\b; ";
      }
      cout << "\b\b] = best = "  << best_f << endl;
    }
    delete op;
  }

  return best;
}

void EGO::latin_hypercube(size_t F, int D)
{
  int* latin = ihs(dimension, F, D, D);
  if(!latin) {
    cout << "Sample plan broke horribly, exiting" << endl;
    exit(-1);
  }
  for(size_t i = 0; i < F; i++) {
    vector<double> x(dimension, 0.0);
    for(int j = 0; j < dimension; j++) {
      double frac = (upper[j] - lower[j]) / (F-1);
      x[j] = lower[j] + frac*(latin[i*dimension+j]-1);
      if(is_discrete) {
        x[j] = floor(x[j]);
      }
    }
    while(running.size() >= num_lambda) {
      //std::this_thread::sleep_for(std::chrono::milliseconds(20));
      check_running_tasks();
    }
    evaluate(x);
  }
}

void EGO::sample_plan(size_t F, int D)
{
  size_t size_latin = floor(F/3);
  cout << size_latin << " size" << endl;
  D += rand() % 5;
  int* latin = ihs(dimension, size_latin, D, D);
  if(!latin) {
    cout << "Sample plan broke horribly, exiting" << endl;
    exit(-1);
  }
  cout << "Latin eval" << endl;
  for(size_t i = 0; i < size_latin; i++) {
    vector<double> x(dimension, 0.0);
    for(int j = 0; j < dimension; j++) {
      double frac = (upper[j] - lower[j]) / (size_latin-1);
      x[j] = lower[j] + frac*(latin[i*dimension+j]-1);
      if(is_discrete) {
        x[j] = floor(x[j]);
      }
    }
    while(running.size() >= num_lambda) {
      //std::this_thread::sleep_for(std::chrono::milliseconds(20));
      //check_running_tasks();
      update_running();
    }
    //evaluate(x);
    python_eval(x);
  }
  while(valid_set.size() < 1 || running.size() >= num_lambda) {
    update_running();
  }
  sg->choose_svm_param(5, true);
  delete latin;

  cout << "Adding extra permutations" << endl;
  for(size_t i = 0; i < F - size_latin; i++) {
    vector<double> point(dimension, 0.0);
    for(int j = 0, loops = 0; j < dimension; loops++, j++) {
      int rand_num = rand() % ((int)round(upper[j] - lower[j]) + 1);
      point[j] = lower[j] + rand_num;
      point[j] = min(point[j], upper[j]);
      point[j] = max(point[j], lower[j]);
      if(j == dimension - 1) {
        if(loops < 1000 && (sg->svm_label(&point[0]) != 1 || has_run(point))) {
	  j = -1; // reset and try and find another random point
	}
      }
    }
    if(sg->svm_label(&point[0]) != 1 || has_run(point)) {
      int choice = 0, valid_size = valid_set.size(), radius = 2;
      if(valid_size > 1) {
        choice = rand() % valid_size;
      }
      int loops = 0;
      for(int j = 0; j < dimension; j++, loops++) {
	//int dist = floor((upper[j] - lower[j]) / radius);
        //point[j] = valid_set[choice][j] + round(uni_dist(0, dist) - dist / 2);
        int rand_num = rand() % (radius + 1);
        point[j] = valid_set[choice][j] + (rand_num - radius / 2);
        if((point[j] > upper[j]) || (point[j] < lower[j])) {
	  j = -1; // reset loop
	}
        if(j == dimension - 1) {
          if(!not_run(&point[0]) || !not_running(&point[0])) {
            j = -1;
          }
        }
	if(loops > 1000) {
	  radius++;
	  cout << loops << endl;
	}
      }
    }
    python_eval(point);
    sg->choose_svm_param(5);

    while(running.size() >= num_lambda) {
      update_running();
    }

  }
  while(running.size() >= num_lambda) {
    update_running();
  }
}

vector<double> EGO::local_random(double radius, int llambda)
{
  if(!suppress) {
    // cout << "Evaluating random pertubations of best results" << endl;
  }
  double noise = sg->error();
  vector<int> index;
  vector<vector<double>> points;
  for(size_t i = 0; i < valid_set.size(); i++) {
    if(abs(training_f[i] - best_fitness) < noise) {
      index.push_back(i);
    }
  }
  vector<double> low_bounds(dimension, -10000000.0);
  double up;
  vector<double> x(dimension, 0.0);
  int num[dimension+1];
  for(size_t i = 0; i < index.size(); i++) {
    num[0] = 1;
    for(int j = 0; j < dimension; j++) {
      double point = valid_set[index[i]][j];
      low_bounds[j] = max(lower[j], point - radius);
      up = min(upper[j], point + radius);
      num[j+1] = num[j] * (int) (up - low_bounds[j] + 1);
      //cout << j << " " <<num[j+1] <<" "<<low_bounds[j]<<" "<<up<< " possible" << endl;
    }
    for(int j = 0; j < num[dimension]; j++) {
      for(int k = 0; k < dimension; k++) {
        x[k] = low_bounds[k] + (j % num[k+1]) / num[k];
	//cout << x[k] << " ";
      }
      //cout << endl;
      if(!has_run(x)) {
	bool can_run = true;
        for(size_t z = 0; z < points.size(); z++) {
	  for(int l = 0; l < dimension; l++) {
	    if(x[l] != points[z][l]) {
	      break;
	    } else if(l == dimension - 1) {
	      can_run = false;
	      break;
	    }
	  }
	  if(!can_run) {
	    break;
	  }
        }
	if(can_run) {
	  points.push_back(x);
	  //cout << "able" << endl;
        }
      }
    }
  }
  int size = points.size();
  if(size > 0) {
    vector<double> result(dimension * llambda, 0.0);
    double choices[llambda];
    for(int i = 0; i < llambda; i++) {
      choices[i] = -1;
      int ind = rand() % size;
      if(size >= llambda) {
        for(int j = 0; j < llambda; j++) {
          if(ind == choices[j]) {
            j = -1;
            ind = rand() % size;
          }
        }
      }
      for(int j = 0; j < dimension; j++) {
        result[i*dimension+j] = points[ind][j];
      }
    }
    return result;
  } else {
    points.clear();
    low_bounds.clear();
    index.clear();
    x.clear();
    return local_random(radius + 1, llambda);
  }
}

vector<double> *EGO::brute_search_swarm(int npts, int llambda, bool use_mean)
{
  double best = 0.001;
  int size = dimension * llambda;
  vector<double> *best_point = new vector<double>(size, 0);
  unsigned long long npts_plus[dimension + 1];
  unsigned long long loop[llambda];
  double steps[dimension];
  bool has_result = false;
  for(int i = 0; i < llambda; i++) loop[i] = i;
  npts_plus[dimension] = 1;
  if(is_discrete && exhaustive) {
    for(int i = dimension - 1; i >= 0; i--) {
      steps[i] = 1.0;
      npts_plus[i] = (upper[i] - lower[i] + 1) * npts_plus[i+1];
    }
  } else {
    for(int i = 0; i < dimension; i++) {
      if(is_discrete) {
        steps[i] = (int) floor((upper[i] - lower[i]) / npts);
        if(steps[i] == 0) steps[i] = 1.0;
      } else {
        steps[i] = (upper[i] - lower[i]) / npts;
      }
      npts_plus[i] = (int) pow(npts + 1, dimension - i);
    }
  }

  if(llambda == 1) {
    vector<double> x(size, 0.0);
    for(unsigned long long i = 0; i < npts_plus[0]; i++) {
      bool can_run = true;
      for(int j = 0; j < dimension; j++) {
        x[j] = lower[j] + floor((i % npts_plus[j]) / npts_plus[j+1]) * steps[j];
        if(x[j] > upper[j] || x[j] < lower[j]) can_run = false;
      }

      if(can_run && not_run(&x[0]) && not_running(&x[0])) {
        if(sg->svm_label(&x[0]) == 1) {
          double result = fitness(x);
          if(result > best) {
            best_point->assign(x.begin(), x.end());
            best = result;
            has_result = true;
          }
	}
      }
    }
  } else {
    pair<double, double> best_mean_var;
    for(int i = 0; i < llambda; i++) {
      best = 0.001;
      vector<double> point((i+1)*dimension, 0.0);
      bool found = false;

      for(int j = 0; j < i*dimension; j++) {
        point[j] = (*best_point)[j];
      }

      for(unsigned long long j = 0; j < npts_plus[0]; j++) {
        bool can_run = true;
	for(int k = 0; k < i; k++) {
	  if(j == loop[k]) {
	    j++;
	    k = 0;
	  }
	}

        for(int k = 0; k < dimension; k++) {
          point[i * dimension + k] = lower[k] + floor((j % npts_plus[k]) / npts_plus[k+1]) * steps[k];
          if(point[i * dimension + k] > upper[k] || point[i * dimension + k] < lower[k]) can_run = false;
        }

        if(can_run && not_run(&point[i*dimension]) && not_running(&point[i*dimension])) {
	  if(sg->svm_label(&point[i*dimension]) == 1) {
	    pair<double, double> p = sg->predict(&point[i*dimension], true);
	    double result = 0.0;
	    if(use_mean) {
	      result = sg->mean(&point[i*dimension]);
	    } else  {
	      result = fitness(point);
		}

            if(result > best) {
              for(int k = 0; k < dimension; k++) {
                (*best_point)[i*dimension + k] = point[i*dimension + k];
              }
	      best_mean_var = p;
              best = result;
	      loop[i] = j;
              if(i == llambda - 1) has_result = true;
	      found = true;
            }
	  }
        }
      }
      if(!found) {
        delete best_point;
        return NULL;
      }
    }
  }

  if(has_result) {
    if(!suppress) {
      cout << "[";
      for(int i = 0; i < llambda; i++) {
        for(int j = 0; j < dimension; j++) {
          cout << (*best_point)[i*dimension + j] << " ";
        }
        cout << "\b; ";
      }
      cout << "\b\b] = best = "  << best << endl;
    }
    return best_point;
  } else {
    delete best_point;
    return NULL;
  }
}


vector<double> EGO::brute_search_local_swarm(const vector<double> &particle, double radius, int llambda, bool has_to_run, bool random, bool use_mean)
{
  double best = 0.001;
  int size = dimension * llambda;
  vector<double> best_point(size, 0);
  unsigned long long loop[llambda];
  double steps[dimension];
  unsigned long long npts_plus[dimension + 1];
  bool has_result = false;
  for(int i = 0; i < llambda; i++) loop[i] = i;
  for(int i = 0; i < dimension; i++) {
    if(is_discrete) {
      steps[i] = 1;
      npts_plus[i] = pow(2*radius + 1, dimension - i);
    } else {
      steps[i] = 2 * radius / max_points;
      npts_plus[i] = pow(max_points + 1, dimension - i);
    }
  }
  npts_plus[dimension] = 1;

  if(llambda == 1) {
      vector<double> x(size, 0.0);
      for(unsigned long long i = 0; i < npts_plus[0]; i++) {
        bool can_run = true;
        for(int j = 0; j < dimension; j++) {
          x[j] = particle[j] + floor(((i % npts_plus[j]) / npts_plus[j+1]) - radius) * steps[j];
          if(x[j] > upper[j] || x[j] < lower[j]) can_run = false;
        }

        if(can_run && (!has_to_run || (not_run(&x[0]) && not_running(&x[0])))) {
	  if(random || sg->svm_label(&x[0]) == 1) {
            double result = fitness(x);
	    if(num_lambda == 1){
	      pair<double, double> p = sg->predict(&x[0], true);
	      result = ei(p.first, p.second, sg->best_raw());
	    }
            if(result > best) {
              best_point = x;
              best = result;
              has_result = true;
            }
	  }
        }
      }
  } else {
    for(int i = 0; i < llambda; i++) {
      best = 0.001;
      vector<double> point((i+1)*dimension, 0.0);
      bool found = false;

      for(int j = 0; j < i*dimension; j++) {
        point[j] = best_point[j];
      }

      for(unsigned long long j = 0; j < npts_plus[0]; j++) {
        bool can_run = true;
	for(int k = 0; k < i; k++) {
	  if(j == loop[k]) {
	    j++;
	    k = 0;
	  }
	}

        for(int k = 0; k < dimension; k++) {
          point[i * dimension + k] = particle[k] + floor(((j % npts_plus[k]) / npts_plus[k+1]) - radius) * steps[k];
          if(point[i * dimension + k] > upper[k] || point[i * dimension + k] < lower[k]) can_run = false;
        }

        if(can_run && (!has_to_run || (not_run(&point[i*dimension]) && not_running(&point[i*dimension])))) {
	    if(random || sg->svm_label(&point[i*dimension]) == 1) {
	    double result = fitness(point);
            if(result > best) {
              best_point = point;
              best = result;
	      loop[i] = j;
              if(i == llambda - 1) has_result = true;
	      found = true;
            }
	  }
        }
      }
      if(!found) {
	break;
      }
    }
  }

  if(has_result) {
    return best_point;
  } else {
    if(is_discrete) {
      if(llambda == 1) {
        if(radius < 3) {
          return brute_search_local_swarm(particle, radius + 1, llambda, has_to_run, random, use_mean);
        } else {
          if(!suppress) cout << "Cannot find new points in direct vicinity of best" << endl;
	  best_point.clear();
          return best_point;
	}
      }
      if(radius / llambda > upper[0] - lower[0]) {
        if(!suppress) cout << "Cannot find new points in direct vicinity of best" << endl;
	best_point.clear();
        return best_point;
      } else {
        //cout << "Increasing radius" << endl;
        return brute_search_local_swarm(particle, radius + 1, llambda, has_to_run, random, use_mean);
      }
    } else {
      if(llambda == 1 && radius < 3) {
	max_points *= 2;
        return brute_search_local_swarm(particle, radius + 1, llambda, has_to_run, random, use_mean);
      } else if(radius / llambda > upper[0] - lower[0]) {
        if(!suppress) cout << "Cannot find new points in direct vicinity of best" << endl;
	best_point.clear();
        return best_point;
      } else {
        cout << "Increasing radius" << endl;
        return brute_search_local_swarm(particle, radius + 1, llambda, has_to_run, random, use_mean);
      }
    }
  }
}

bool EGO::has_run(const vector<double> &point)
{
  return !(not_run(&point[0]) && not_running(&point[0]));
}

bool EGO::not_run(const double x[])
{
  static double eps = 0.001;
  vector<vector<double>>::iterator train = training.begin();
  while(train != training.end()) {
    int i = 0;
    for(; i < dimension; i++) {
      if(abs((*train)[i] - x[i]) > eps) break;
    }
    if(i == dimension) return false;
    train++;
  }
  return true;
}

bool EGO::not_running(const double x[])
{
  static double eps = 0.001;
  vector<struct running_node>::iterator node = running.begin();
  while(node != running.end()) {
    int i = 0;
    for(; i < dimension; i++) {
      if(abs(node->data[i] - x[i]) > eps) break;
    }
    if(i == dimension) return false;
    node++;
  }
  //Not in training or running set, so hasn't been run
  return true;

}

double EGO::ei(double y, double var, double y_min) {
	if (var <= 0.0) {
		return 0.0;
	}

	double sd = sqrt(var);
	double y_diff = y_min - y;
	double y_diff_s = y_diff / sd;
	return y_diff * gsl_cdf_ugaussian_P(y_diff_s) + sd * gsl_ran_ugaussian_pdf(y_diff_s);
}

double EGO::ei_multi(double lambda_s2[], double lambda_mean[], int max_lambdas, int n, double y_best)
{
    double sum_ei=0.0, e_i=0.0;
    int max_mus = mu_means.size();

      for (int k=0; k < n; k++) {
          double min = y_best;
          for(int i=0; i < max_mus; i++){
              double mius = gsl_ran_ugaussian(rng) * mu_vars[i] + mu_means[i];
              if (mius < min)
                  min = mius;
          }
          double min2=100000000.0;
          for(int j=0;j<max_lambdas;j++){
              double l = gsl_ran_ugaussian(rng) * lambda_s2[j] + lambda_mean[j];
              if (l < min2){
                  min2 = l;
	      }
          }

          e_i = min - min2;
          if (e_i < 0.0) {
            e_i = 0.0;
          }
          sum_ei = e_i + sum_ei;
      }
    return sum_ei;
}

void EGO::update_best_result(vector<double> x, double y) {
	if (y < best_fitness) {
		best_particle = x;
		best_fitness = y;
	}
}
