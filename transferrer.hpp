#pragma once

#include <string>
#include <vector>
#include "evaluator.hpp"
#include "functions.hpp"
#include "gsl_rng.h"

using namespace std;

class Transferrer {
	public:
		Transferrer(
			results_t&_results_old,
			results_t& results_new,
			Evaluator& evaluator,
			config_t config);
		~Transferrer();
		void run();

	private:
		Evaluator& evaluator;
		config_t config;
		size_t max_trials;
		double sig_level;
		boundaries_t boundaries;
		bool is_discrete;
		double fitness_percentile;
		results_t results_old;
		results_t results_new;
		gsl_rng* rng;

		results_t sample_results_old();
		static bool fitness_more_than(
			pair<vector<double>, vector<double>> xs,
			pair<vector<double>, vector<double>> ys);
		double calc_fitness_percentile(double percentile);
};
