#include "optimise.h"

void opt::update_particles(int generation, int max_iter)
{
  double frac = generation / max_iter;
  uniform_real_distribution<> dist(0, 2.0);
  random_device rd;
  mt19937 gen(rd());

  for(Particle *part : particles) {
    for(int j = 0; j < dimension; j++) {
      double maxVel = (1 - pow(frac, 2.0)) * speed_max[j];
      part->speed[j] += dist(gen) * (best_part->p[j] - part->p[j]);
      part->speed[j] += dist(gen) * (part->best[j] - part->p[j]);

      if (part->speed[j] < -maxVel) part->speed[j] = -maxVel;
      if (part->speed[j] > maxVel) part->speed[j] = maxVel;

      part->p[j] += part->speed[j];
    }
  }
}

void opt::filter()
{
  for(int i = 0; i < particles.size(); i++) {
    Particle *part = particles[i];
    for(int j = 0; j < dimension; j++) {
      if(is_discrete) {
        part->p[j] = round(part->p[j]);
      }
      part->p[j] = min(part->p[j], upper[j]);
      part->p[j] = max(part->p[j], lower[j]);
      if(part->p[j] == 0.0) part->p[j] = 0.0;
    }
  }
}

opt::opt(int d, vector<double> u, vector<double> l, EGO *e, bool disc)
{
  dimension = d;
  upper = u;
  lower = l;
  ego = e;
  is_discrete = disc;
  best_part = new Particle();

  space_generator = new vector<uniform_real_distribution<>>();
  speed_generator = new vector<uniform_real_distribution<>>();
  speed_max = vector<double>(dimension, 0.0);

  for(int i = 0; i < dimension; i++) {
    uniform_real_distribution<> space_dist(lower[i], upper[i]);

    speed_max[i] = 0.2 * (upper[i] - lower[i]);
    uniform_real_distribution<> speed_dist(-speed_max[i], speed_max[i]);

    space_generator->push_back(space_dist);
    speed_generator->push_back(speed_dist);
  }
}

void opt::generate(int pop)
{
  random_device rd;
  mt19937 gen(rd());
  for(int i = 0; i < pop; i++) {
    Particle *part = new Particle();
    for(int j = 0; j < dimension; j++) {
      part->p.push_back((*space_generator)[j](gen));
      part->best.push_back(part->p[j]);
      part->speed.push_back((*speed_generator)[j](gen));
    }
    part->best_fitness = 100000000;
    particles.push_back(part);
  }
  best_part = new Particle();
  for(int j = 0; j < dimension; j++) {
    best_part->p.push_back((*space_generator)[j](gen));
  }
  best_part->best_fitness = 1000000000;
  filter();
}

vector<double> opt::swarm_optimise(vector<double> best, int max_gen, int pop)
{
  generate(pop);
  Particle *part = particles[0];
  for(int i = 0; i < dimension; i++) {
   part->p[i] = best[i];
  }
  return swarm_main_optimise(max_gen);
  
}

vector<double> opt::swarm_optimise(int max_gen, int pop)
{
  generate(pop);
  return swarm_main_optimise(max_gen);
}

vector<double> opt::swarm_main_optimise(int max_gen)
{
  for(int g = 0; g < max_gen; g++) {
    for(int i = 0; i < particles.size(); i++) {
      Particle *part = particles[i];
      double result = ego->fitness(part->p);

      if(result < part->best_fitness) {
        part->best_fitness = result;
	part->best = part->p;
      }

      if(result < best_part->best_fitness) {
        best_part->best_fitness = result;
	best_part->p = part->p;
      }
    }
    update_particles(g, max_gen);
    filter();
  }

  return best_part->p;
}
