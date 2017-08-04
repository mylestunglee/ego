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

		void read_results(string filename);
		results_t sample_results_old();
		double calc_label_correlation(results_t results_new);
		static bool fitness_more_than(
			pair<vector<double>, vector<double>> xs,
			pair<vector<double>, vector<double>> ys);
		void interpolate(boundaries_t boundaries_old, results_t results_new,
			results_t predictions);
		vector<double> test_correlation(vector<double> xs, vector<double> ys);
		double calc_fitness_percentile(double percentile);
		results_t transfer_results_old(GaussianProcess& surrogate, results_t sampled);
};
