#include <vector>
#include <random>
#include "surrogate.hpp"
#include <mutex>
#include "evaluator.hpp"
#include <gsl/gsl_rng.h>
#include <gsl/gsl_vector.h>
#pragma once

using namespace std;

class EGO {
	public:
		EGO(vector<pair<double, double>> boundaries, Evaluator& evaluator);
		~EGO();
		void sample_latin(size_t n);
		void sample_uniform(size_t n);
		void run();
	private:
		size_t dimension;
		size_t max_evaluations;
		size_t evaluations;
		size_t max_trials;
		double convergence_threshold;
		vector<double> best_particle;
		double best_fitness;

		Evaluator& evaluator;

		mutex evaluator_lock;
		vector<pair<double, double>> boundaries;

		Surrogate* sg;
		Surrogate* sg_label;
		Surrogate* sg_cost;

		gsl_rng* rng;

		vector<double> maximise_expected_improvement(double &improvement);
		static double expected_improvement_bounded(const gsl_vector* v, void* p);
		static double expected_improvement(double y, double sd, double y_min);
		static void thread_evaluate(EGO* ego, vector<double> x);
		void evaluate(vector<vector<double>> xs);
};
