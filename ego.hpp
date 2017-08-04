#include <vector>
#include <random>
#include "gp.hpp"
#include <mutex>
#include "evaluator.hpp"
#include "functions.hpp"
#include <gsl/gsl_rng.h>
#include <gsl/gsl_vector.h>
#pragma once

using namespace std;

class EGO {
	public:
		EGO(
			Evaluator& evaluator,
			boundaries_t boundaries,
			boundaries_t rejection,
			size_t max_evaluations,
			size_t max_trials,
			double convegence_threshold,
			bool is_discrete,
			size_t constraints,
			size_t costs);
		~EGO();
		void sample_latin(size_t n);
		void sample_uniform(size_t n);
		void run();
		void simulate(vector<double> x, vector<double> y);
	private:
		size_t dimension;
		size_t max_evaluations;
		size_t evaluations;
		size_t max_trials;
		double convergence_threshold;
		bool is_discrete;
		vector<double> x_opt;
		double y_opt;

		Evaluator& evaluator;

		mutex evaluator_lock;
		boundaries_t boundaries;
		boundaries_t rejection;

		Surrogate* sg;
		Surrogate* sg_label;
		vector<Surrogate*> constraints;
		vector<Surrogate*> costs;

		gsl_rng* rng;

		static double expected_improvement_bounded(const gsl_vector* v, void* p);
		static double expected_improvement(double y, double sd, double y_min);
		static void thread_evaluate(EGO* ego, vector<double> x);
		void evaluate(vector<vector<double>> xs);
		double predict_cost(vector<double> x);
		double success_constraints_probability(vector<double> x);
		static vector<double> generate_random_point(void* p);
};
