#include "ego.hpp"
#include "surrogate.hpp"
#include "optimise.hpp"
#include "evaluator.hpp"

using namespace std;

int main(int argc, char * argv[]) {

	vector<double> lower = {-2, -2};
	vector<double> upper = {2, 2};
	int dimension = 2;

	Surrogate *sg = new Surrogate(dimension, SEiso, true, true);
	Evaluator* evaluator = new Evaluator("./test_script");

	EGO *ego = new EGO(dimension, sg, lower, upper, "", 1);
	ego->evaluator = evaluator;
	ego->search_type = 2; // PSO
	ego->use_cost = 1; //EI/cost

	ego->sample_plan(3 * dimension, 5);
	ego->run_quad();

	evaluator->save("fitness.log");

	delete sg;
	delete ego;
	delete evaluator;
}
