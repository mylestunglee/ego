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
		int dimension;
		int max_iterations;
		int num_iterations;
		vector<double> best_particle;
		long double best_fitness;
		bool is_discrete;
		bool suppress;

		Evaluator& evaluator;

		mutex evaluator_lock;
	vector<pair<double, double>> boundaries;

		Surrogate* sg;
		Surrogate* sg_cost;

		gsl_rng* rng;

		vector<double> maximise_expected_improvement();
		static double expected_improvement(const gsl_vector* v, void* p);
		static double ei(double y, double var, double y_min);
		static void thread_evaluate(EGO* ego, vector<double> x);
		void evaluate(vector<vector<double>> xs);
};
