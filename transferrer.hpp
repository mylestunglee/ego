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
			string filename_results_old,
			Evaluator& evaluator,
			size_t max_evaluations,
			size_t max_trials,
			double convergence_threshold,
			double sig_level,
			boundaries_t boundaries,
			bool is_discrete,
			size_t constraints,
			size_t costs,
			double fitness_percentile);
		~Transferrer();
		void run();

	private:
		Evaluator& evaluator;
		size_t max_evaluations;
		size_t max_trials;
		double convergence_threshold;
		double sig_level;
		boundaries_t boundaries;
		bool is_discrete;
		size_t constraints;
		size_t costs;
		double fitness_percentile;
		results_t results_old;
		gsl_rng* rng;

		results_t sample_results_old();
		static bool fitness_more_than(
			pair<vector<double>, vector<double>> xs,
			pair<vector<double>, vector<double>> ys);
		double calc_fitness_percentile(double percentile);
};
