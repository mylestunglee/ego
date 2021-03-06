#pragma once
#include <set>
#include <vector>
#include <mutex>
#include <gsl_rng.h>
#include <gsl_vector.h>
#include "gp.hpp"
#include "evaluator.hpp"
#include "functions.hpp"

using namespace std;

class EGO {
	public:
		EGO(
			Evaluator& evaluator,
			config_t config,
			boundaries_t rejection = {},
			results_t results_old = {});
		~EGO();
		void sample_latin(size_t n);
		void sample_uniform(size_t n);
		void run();
		void simulate(vector<double> x, vector<double> y);
	private:
		size_t dimension;
		boundaries_t boundaries;
		boundaries_t rejection;
		double budget;
		double accum_cost;
		size_t max_trials;
		double convergence_threshold;
		bool is_discrete;
		vector<double> x_opt;
		double y_opt;

		Evaluator& evaluator;

		mutex evaluator_lock;

		Surrogate* fitness;
		Surrogate* label;
		vector<Surrogate*> constraints;
		vector<Surrogate*> costs;

		gsl_rng* rng;

		static double expected_improvement_bounded(const gsl_vector* v, void* p);
		static double expected_improvement(double y, double sd, double y_min);
		static void thread_evaluate(EGO* ego, vector<double> x);
		void evaluate(vector<vector<double>> xs);
		double predict_cost(vector<double> x);
		double result_cost(vector<double> y);
		double success_constraints_probability(vector<double> x);
		static vector<double> generate_random_point(void* p);
};
