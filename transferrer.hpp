#include <string>
#include <vector>
#include "evaluator.hpp"
#include "functions.hpp"
#include "surrogate.hpp"

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
			size_t costs);
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
		results_t results_old;
		boundaries_t space_intersection;
		Surrogate* predictor;
		boundaries_t space_extend;
		gsl_rng* rng;

		void read_results(string filename);
		results_t sample_results_old();
		double calc_label_correlation(results_t results_new);
		static bool fitness_more_than(
			pair<vector<double>, vector<double>> xs,
			pair<vector<double>, vector<double>> ys);
		void interpolate(boundaries_t boundaries_old, vector<double> coeffs,
			results_t results_new);
		void extrude(boundaries_t boundaries_old);
		void reduce(boundaries_t boundaries_old);
		static vector<double> generate_random_point(void* p);
		static double cross_section_correlation(const gsl_vector* v, void* p);
		vector<double> test_correlation(vector<double> xs, vector<double> ys);
		double calc_fitness_percentile(double percentile);
};
