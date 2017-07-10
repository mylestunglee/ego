#include <string>
#include <vector>
#include "evaluator.hpp"
#include "functions.hpp"

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
		void run();
	private:
		vector<pair<vector<double>, vector<double>>> results_old;
		Evaluator& evaluator;
		size_t max_evaluations;
		size_t max_trials;
		double convergence_threshold;
		double sig_level;
		boundaries_t boundaries;
		bool is_discrete;
		size_t constraints;
		size_t costs;

		void read_results(string filename);
		results_t sample_results_old();
		double calc_label_correlation(results_t results_new);
		static bool fitness_more_than(
			pair<vector<double>, vector<double>> xs,
			pair<vector<double>, vector<double>> ys);
		void interpolate(boundaries_t boundaries_old, vector<double> coeffs,
			results_t results_new);

};
