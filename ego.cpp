#include <limits>
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
	assert(num_iterations > 0);

	sg->train_gp_first();
	while(num_iterations < max_iterations) {
		sg->train();
		sg_cost->train();

		cout << "Iteration: " << num_iterations << endl;
		evaluate(group(max_ei_par(lambda), dimension));
	}
}

vector<double> EGO::best_result()
{
  return best_particle;
}

double EGO::fitness(const vector<double> &x) {
	int num = x.size() / dimension;

	double lambda_means[num];
	double lambda_vars[num];
	double cost = 1.0, mean_cost=1.0;

	for(int i = 0; i < num; i++) {
		double y [dimension];

		for(int j = 0; j < dimension; j++) {
			y[j] = x[i * dimension + j];
		}

		pair<double, double> p = sg->predict(y, true);
		lambda_means[i] = p.first;
		lambda_vars[i] = p.second;

		pair<double, double> p_cost = sg_cost->predict(y);
		cost += p_cost.first;
		mean_cost = sg_cost->mean_fit;
	}

  double result = ei_multi(lambda_vars, lambda_means, num, n_sims, sg->best_raw());
  if(cost > 0) {
    result *= mean_cost / cost;
  }
  return exp(abs(result/n_sims) - 1) * 100;
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

void EGO::sample_plan(size_t n)
{
	int seed = gsl_rng_get(rng);
	int* latin = ihs(dimension, n, 5, seed);
	assert(latin != NULL);

	vector<vector<double>> xs;

	// Scale latin hypercube to fit parameter space
	for (size_t i = 0; i < n; i++) {
		vector<double> x;
		for (size_t j = 0; j < (unsigned) dimension; j++) {
			double x_j = lower[j] + (latin[i * dimension + j] - 1) * (n - 1) * (upper[j] - lower[j]);
			if (is_discrete) {
				x_j = round(x_j);
			}
			x.push_back(x_j);
		}
		xs.push_back(x);
	}

	delete latin;

	evaluate(xs);

	sg->choose_svm_param(5, true);
}

void EGO::uniform_sample(size_t n) {
	for (size_t i = 0; i < n; i++) {
		vector<double> x(dimension, 0.0);
		for (size_t trial = 0; trial < 30; trial++) {
			// Sample parameter space using uniform distribution
			for (int j = 0; j < dimension; j++) {
				x[j] = gsl_ran_flat(rng, lower[j], upper[j]);
			}

			// Predicted label is valid
			if (sg->svm_label(&x[0]) == 1) {
				evaluate({x});
				sg->choose_svm_param(5);

				// Find next point
				break;
			}
		}
	}
}

vector<double> EGO::local_random(double radius, int llambda) {
	if (!suppress) {
		// cout << "Evaluating random pertubations of best results" << endl;
	}
	double noise = sg->error();
	vector<int> index;
	vector<vector<double>> points;
	for (size_t i = 0; i < training_f.size(); i++) {
		if (abs(training_f[i] - best_fitness) < noise) {
			index.push_back(i);
		}
	}

	vector <double> low_bounds(dimension, -10000000.0);
	double up;
	vector <double> x(dimension, 0.0);
	int num[dimension + 1];
	for (size_t i = 0; i < index.size(); i++) {
		num[0] = 1;
		for (int j = 0; j < dimension; j++) {
			double point = valid_set[index[i]][j];
			low_bounds[j] = max(lower[j], point - radius);
			up = min(upper[j], point + radius);
			num[j + 1] = num[j] * (int)(up - low_bounds[j] + 1);
			//cout << j << " " <<num[j+1] <<" "<<low_bounds[j]<<" "<<up<< " possible" << endl;
		}
		for (int j = 0; j < num[dimension]; j++) {
			for (int k = 0; k < dimension; k++) {
				x[k] = low_bounds[k] + (j % num[k + 1]) / num[k];
				//cout << x[k] << " ";
			}
			bool can_run = true;
			for (size_t z = 0; z < points.size(); z++) {
				for (int l = 0; l < dimension; l++) {
					if (x[l] != points[z][l]) {
						break;
					} else if (l == dimension - 1) {
						can_run = false;
						break;
					}
				}
				if (!can_run) {
					break;
				}
			}
			if (can_run) {
				points.push_back(x);
			}
		}
	}
	int size = points.size();
	if (size > 0) {
		vector < double > result(dimension * llambda, 0.0);
		double choices[llambda];
		for (int i = 0; i < llambda; i++) {
			choices[i] = -1;
			int ind = rand() % size;
			if (size >= llambda) {
				for (int j = 0; j < llambda; j++) {
					if (ind == choices[j]) {
						j = -1;
						ind = rand() % size;
					}
				}
			}
			for (int j = 0; j < dimension; j++) {
				result[i * dimension + j] = points[ind][j];
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

      if(can_run) {
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

        if(can_run) {
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

        if(can_run) {
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

        if(can_run) {
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

double EGO::ei(double y, double var, double y_min) {
	if (var <= 0.0) {
		return 0.0;
	}

	double sd = sqrt(var);
	double y_diff = y_min - y;
	double y_diff_s = y_diff / sd;
	return y_diff * gsl_cdf_ugaussian_P(y_diff_s) + sd * gsl_ran_ugaussian_pdf(y_diff_s);
}

double EGO::ei_multi(double lambda_s2[], double lambda_mean[], int max_lambdas, int n, double y_best) {
	double result = 0.0;

	for (int k=0; k < n; k++) {
		double minimum = numeric_limits<double>::max();
		for(int j=0;j<max_lambdas;j++){
			minimum = min(gsl_ran_ugaussian(rng) * lambda_s2[j] + lambda_mean[j], minimum);
		}

		double ei = y_best - minimum;

		if (ei > 0.0) {
			result += ei;
		}
	}
	return result;
}

void EGO::update_best_result(vector<double> x, double y) {
	if (y < best_fitness) {
		best_particle = x;
		best_fitness = y;
	}
}

/* Evaluates a vector to add to the training set */
void EGO::evaluate2(EGO* ego, vector<double> x) {
	assert(ego != NULL);

	vector<double> y = ego->evaluator.evaluate(x);

	ego->running_mtx.lock();

    ego->num_iterations++;

	ego->sg->add(x, y[0], y[1] == 0 ? 1 : 2, 0);
	ego->sg_cost->add(x, y[2]);

	if (y[1] == 0) {
		ego->valid_set.push_back(x);
		ego->update_best_result(x, y[0]);
		ego->training_f.push_back(y[0]);
	}

	ego->training.push_back(x);

	ego->running_mtx.unlock();
}

/* Concurrently evaluates multiple points xs */
void EGO::evaluate(vector<vector<double>> xs) {
	vector<thread> threads;

	for (auto x : xs) {
		threads.push_back(thread(evaluate2, this, x));
	}

	for (auto& t : threads) {
		t.join();
	}
}

/* Group sequences of elements into vectors of size size */
vector<vector<double>> EGO::group(vector<double> xs, size_t size) {
	assert(xs.size() % size == 0);

	vector<vector<double>> result(xs.size() / size, vector<double>());

	for (size_t i = 0; i < xs.size(); i++) {
		result[i / size].push_back(xs[i]);
	}

	return result;
}
