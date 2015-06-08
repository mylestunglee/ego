

vector<double> EGO::brute_search_local_loop(vector<double> particle, int npts, double radius, int lambda, bool has_to_run)
{
  double best = 1000000;
  int size = dimension * lambda;
  vector<double> best_point(size, 0);
  int loop[lambda];
  double steps[dimension];
  bool has_result = false;
  bool more_viable = true;
  for(int i = 0; i < lambda; i++) loop[i] = i;
  for(int i = 0; i < dimension; i++) {
    if(is_discrete) {
      steps[i] = (int) floor(2 * radius / npts);
      if(steps[i] == 0) steps[i] = 1;
    } else {
      steps[i] = 2 * radius / npts;
    }
  }

  if(lambda == 1) {
    while(more_viable) {
      vector<double> x(size, 0.0);
      bool can_run = true;
      for(int i = 0; i < lambda; i++) {
        for(int j = 0; j < dimension; j++) {
          x[i * dimension + j] = particle[j] + (floor((loop[i] % (int) pow(npts + 1, dimension - j)) / pow(npts + 1, dimension - j - 1)) - npts/2) * steps[j];
          if(x[j] > upper[j % dimension] || x[j] < lower[j % dimension]) can_run = false;
	}
      }
      int i = lambda - 1;
      for(; i >= 0; i--) {
        if(++loop[i] == (pow(npts + 1, dimension) + i - (lambda - 1))) {
          if(i == 0) more_viable = false;
        } else {
          break;
        }
      }
      for(int j = max(i + 1, 1); j < lambda; j++) {
        loop[j] = loop[j-1] + 1;
	if(loop[i] >= pow(npts + 1, dimension)) more_viable = false;
      }

      if(can_run && (!has_to_run || (not_run(&x[0]) && not_running(&x[0])))) {
        double mean = sg->mean(&x[0]);
        double var = sg->var(&x[0]);
        double result = -ei(mean, var, best_fitness);
        if(result < best) {
          best_point = x;
          best = result;
          has_result = true;
        }
      }
    }
  } else {
    //lambda >= 2
    while(more_viable) {
      vector<double> x(size, 0.0);
      bool can_run = true;
      for(int i = 0; i < lambda; i++) {
        for(int j = 0; j < dimension; j++) {
          x[i * dimension + j] = particle[j] + (floor((loop[i] % (int) pow(npts + 1, dimension - j)) / pow(npts + 1, dimension - j - 1)) - npts/2) * steps[j];
          if(x[i*dimension +j] > upper[j] || x[i*dimension+j] < lower[j]) can_run = false;
	}
	if(!can_run) break;
      }
      int i = lambda - 1;
      for(; i >= 0; i--) {
        if(++loop[i] == (pow(npts + 1, dimension) + i - (lambda - 1))) {
          if(i == 0) more_viable = false;
        } else {
          break;
        }
      }
      for(int j = max(i + 1, 1); j < lambda; j++) {
        loop[j] = loop[j-1] + 1;
	if(loop[j] >= pow(npts + 1, dimension) + j - (lambda - 1)) more_viable = false;
      }

      if(can_run && (!has_to_run || (not_run(&x[0]) && not_running(&x[0])))) {
        double result = fitness(x);
        if(result < best) {
          best_point = x;
	  best = result;
	  has_result = true;
        }
      }
    }
  }
  if(has_result) {
    return best_point;
  } else if(has_to_run) {
    return brute_search_local_loop(particle, 2*(radius + 1), radius + 1, lambda, has_to_run);
  } else {
    return brute_search_local_loop(particle, npts, radius + 1, lambda, has_to_run);
  }
}

vector<double>* EGO::brute_search_loop(int npts, int lambda, double min_ei)
{
  double best = 1000000;
  int size = dimension * lambda;
  vector<double> *best_point = new vector<double>(size, 0);
  int loop[lambda];
  double steps[dimension];
  bool has_result = false;
  bool more_viable = true;
  for(int i = 0; i < lambda; i++) loop[i] = i;
  for(int i = 0; i < dimension; i++) {
    if(is_discrete) {
      steps[i] = (int) floor((upper[i] - lower[i]) / npts);
      if(steps[i] == 0) steps[i] = 1;
    } else {
      steps[i] = (upper[i] - lower[i]) / npts;
    }
  }

  if(lambda == 1) {
    while(more_viable) {
      vector<double> x(size, 0.0);
      bool can_run = true;
      for(int i = 0; i < lambda; i++) {
        for(int j = 0; j < dimension; j++) {
          x[i * dimension + j] = lower[j] + floor((loop[i] % (int) pow(npts + 1, dimension - j)) / pow(npts + 1, dimension - j - 1)) * steps[j];
          if(x[i*dimension +j] > upper[j] || x[i*dimension+j] < lower[j]) can_run = false;
	}
      }
      int i = lambda - 1;
      for(; i >= 0; i--) {
        if(++loop[i] == (pow(npts + 1, dimension) + i - (lambda - 1))) {
          if(i == 0) more_viable = false;
        } else {
          break;
        }
      }
      for(int j = max(i + 1, 1); j < lambda; j++) {
        loop[j] = loop[j-1] + 1;
	if(loop[i] >= pow(npts + 1, dimension)) more_viable = false;
      }

      if(can_run) {
        double mean = sg->mean(&x[0]);
        double var = sg->var(&x[0]);
        double result = -ei(mean, var, best_fitness);
        if(result < min(best, -min_ei)) {
          best_point->assign(x.begin(), x.end());
          best = result;
          has_result = true;
        }
      }
    }
  } else {
    //lambda >= 2
    int num_loops = pow(npts + 1, dimension);
    while(more_viable) {
      vector<double> x(size, 0.0);
      bool can_run = true;
      for(int i = 0; i < lambda; i++) {
        for(int j = 0; j < dimension; j++) {
          x[i * dimension + j] = lower[j] + floor((loop[i] % (int) pow(npts + 1, dimension - j)) / pow(npts + 1, dimension - j - 1)) * steps[j];
          if(x[j] > upper[j % dimension] || x[j] < lower[j % dimension]) can_run = false;
	}
      }
      int i = lambda - 1;
      for(; i >= 0; i--) {
        if(++loop[i] == (num_loops + i - (lambda - 1))) {
          if(i == 0) more_viable = false;
        } else {
          break;
        }
      }
      for(int j = max(i + 1, 1); j < lambda; j++) {
        loop[j] = loop[j-1] + 1;
	if(loop[j] >= num_loops) more_viable = false;
      }

      if(can_run) {
        double result = fitness(x);
        if(result < min(best, -min_ei)) {
          best_point->assign(x.begin(), x.end());
	  best = result;
	  has_result = true;
        }
      }
    }
  }

  if(has_result) {
    if(!suppress) {
      cout << "["; for(int i = 0; i < lambda; i++) {
        for(int j = 0; j < dimension; j++) {
          cout << (*best_point)[i*dimension + j] << " ";
        }
        cout << "\b; ";
      }
      cout << "\b\b] = best = "  << best << endl;
    }
    return best_point;
  } else {
    delete best_point;
    return NULL;
  }
}

vector<double> EGO::brute_search(int npts=10, int lambda=1)
{
  double best = 1000000;
  int size = dimension * lambda;
  vector<double> best_point(size, 0);
  double points[size][npts + 1];
  int num_steps = npts + 1;
  bool has_result = false;

  if(lambda == 1) {
    //Build our steps of possible values in each dimension
    for(int i = 0; i < dimension; i++) {
      if(is_discrete) {
        int step = floor((upper[i] - lower[i]) / npts);
        if(step == 0) step = 1;
        int j = 0;
        for(; j <= npts && (lower[i] + j * step) <= upper[i]; j++) {
          points[i][j] = floor(lower[i] + j * step);
        }
        num_steps = min(num_steps, j);
      } else {
        double step = (upper[i] - lower[i]) / npts;
        int j = 0;
        for(; j <= npts; j++) {
          points[i][j] = lower[i] + j * step;
        }
        num_steps = min(num_steps, j);
      }
    }

    //Loop through each possible combination of values
    for(int i = 0; i < pow(num_steps, dimension); i++) {
      double x[dimension];
      for(int j = 0; j < dimension; j++) {
        x[j] = points[j][((int)(i / pow(num_steps, j))) % num_steps];
      }
      if(not_running(x)) {
        double result = -ei(sg->mean(x), sg->var(x), best_fitness);
        if(result < best) {
          best = result;
          best_point.assign(x, x + dimension);
          has_result = true;
        }
      }
    }
  } else {
    //Build our steps of possible values in each dimension
    for(int i = 0; i < size; i++) {
      if(is_discrete) {
        int step = floor((upper[i % dimension] - lower[i % dimension]) / npts);
        if(step == 0) step = 1;
        int j = 0;
        for(; j <= npts && (lower[i % dimension] + j * step) <= upper[i % dimension]; j++) {
          points[i][j] = floor(lower[i % dimension] + j * step);
        }
        num_steps = min(num_steps, j);
      } else {
        double step = (upper[i % dimension] - lower[i % dimension]) / npts;
        int j = 0;
        for(; j <= npts; j++) {
          points[i][j] = lower[i % dimension] + j * step;
        }
        num_steps = min(num_steps, j);
      }
    }

    //Loop through each possible combination of values
    for(int i = 0; i < pow(num_steps, size); i++) {
      vector<double> x(size, 0.0);
      for(int j = 0; j < size; j++) {
        x[j] = points[j][((int) floor(i / pow(num_steps, j))) % num_steps];
      }
      double result = fitness(x);
      if(result < best) {
        best = result;
        best_point = x;
        has_result = true;
      }
    }
  }

  if(has_result) { 
    return best_point;
  } else {
    if(num_steps > npts || !is_discrete) {
      return brute_search(npts * 2, lambda);
    }
    cout << "Broken, can't brute search" << endl;
    exit(1);
  }
}

vector<double> EGO::brute_search_local_loop(vector<double> particle, int npts, double radius, int lambda, bool has_to_run)
{
  double best = 1000000;
  int size = dimension * lambda;
  vector<double> best_point(size, 0);
  int loop[lambda];
  double steps[dimension];
  bool has_result = false;
  bool more_viable = true;
  for(int i = 0; i < lambda; i++) loop[i] = i;
  for(int i = 0; i < dimension; i++) {
    if(is_discrete) {
      steps[i] = (int) floor(2 * radius / npts);
      if(steps[i] == 0) steps[i] = 1;
    } else {
      steps[i] = 2 * radius / npts;
    }
  }

  if(lambda == 1) {
    while(more_viable) {
      vector<double> x(size, 0.0);
      bool can_run = true;
      for(int i = 0; i < lambda; i++) {
        for(int j = 0; j < dimension; j++) {
          x[i * dimension + j] = particle[j] + (floor((loop[i] % (int) pow(npts + 1, dimension - j)) / pow(npts + 1, dimension - j - 1)) - npts/2) * steps[j];
          if(x[j] > upper[j % dimension] || x[j] < lower[j % dimension]) can_run = false;
	}
      }
      int i = lambda - 1;
      for(; i >= 0; i--) {
        if(++loop[i] == (pow(npts + 1, dimension) + i - (lambda - 1))) {
          if(i == 0) more_viable = false;
        } else {
          break;
        }
      }
      for(int j = max(i + 1, 1); j < lambda; j++) {
        loop[j] = loop[j-1] + 1;
	if(loop[i] >= pow(npts + 1, dimension)) more_viable = false;
      }

      if(can_run && (!has_to_run || (not_run(&x[0]) && not_running(&x[0])))) {
        double mean = sg->mean(&x[0]);
        double var = sg->var(&x[0]);
        double result = -ei(mean, var, best_fitness);
        if(result < best) {
          best_point = x;
          best = result;
          has_result = true;
        }
      }
    }
  } else {
    //lambda >= 2
    while(more_viable) {
      vector<double> x(size, 0.0);
      bool can_run = true;
      for(int i = 0; i < lambda; i++) {
        for(int j = 0; j < dimension; j++) {
          x[i * dimension + j] = particle[j] + (floor((loop[i] % (int) pow(npts + 1, dimension - j)) / pow(npts + 1, dimension - j - 1)) - npts/2) * steps[j];
          if(x[i*dimension +j] > upper[j] || x[i*dimension+j] < lower[j]) can_run = false;
	}
	if(!can_run) break;
      }
      int i = lambda - 1;
      for(; i >= 0; i--) {
        if(++loop[i] == (pow(npts + 1, dimension) + i - (lambda - 1))) {
          if(i == 0) more_viable = false;
        } else {
          break;
        }
      }
      for(int j = max(i + 1, 1); j < lambda; j++) {
        loop[j] = loop[j-1] + 1;
	if(loop[j] >= pow(npts + 1, dimension) + j - (lambda - 1)) more_viable = false;
      }

      if(can_run && (!has_to_run || (not_run(&x[0]) && not_running(&x[0])))) {
        double result = fitness(x);
        if(result < best) {
          best_point = x;
	  best = result;
	  has_result = true;
        }
      }
    }
  }
  if(has_result) {
    return best_point;
  } else if(has_to_run) {
    return brute_search_local_loop(particle, 2*(radius + 1), radius + 1, lambda, has_to_run);
  } else {
    return brute_search_local_loop(particle, npts, radius + 1, lambda, has_to_run);
  }
}

vector<double>* EGO::brute_search_loop(int npts, int lambda, double min_ei)
{
  double best = 1000000;
  int size = dimension * lambda;
  vector<double> *best_point = new vector<double>(size, 0);
  int loop[lambda];
  double steps[dimension];
  bool has_result = false;
  bool more_viable = true;
  for(int i = 0; i < lambda; i++) loop[i] = i;
  for(int i = 0; i < dimension; i++) {
    if(is_discrete) {
      steps[i] = (int) floor((upper[i] - lower[i]) / npts);
      if(steps[i] == 0) steps[i] = 1;
    } else {
      steps[i] = (upper[i] - lower[i]) / npts;
    }
  }

  if(lambda == 1) {
    while(more_viable) {
      vector<double> x(size, 0.0);
      bool can_run = true;
      for(int i = 0; i < lambda; i++) {
        for(int j = 0; j < dimension; j++) {
          x[i * dimension + j] = lower[j] + floor((loop[i] % (int) pow(npts + 1, dimension - j)) / pow(npts + 1, dimension - j - 1)) * steps[j];
          if(x[i*dimension +j] > upper[j] || x[i*dimension+j] < lower[j]) can_run = false;
	}
      }
      int i = lambda - 1;
      for(; i >= 0; i--) {
        if(++loop[i] == (pow(npts + 1, dimension) + i - (lambda - 1))) {
          if(i == 0) more_viable = false;
        } else {
          break;
        }
      }
      for(int j = max(i + 1, 1); j < lambda; j++) {
        loop[j] = loop[j-1] + 1;
	if(loop[i] >= pow(npts + 1, dimension)) more_viable = false;
      }

      if(can_run) {
        double mean = sg->mean(&x[0]);
        double var = sg->var(&x[0]);
        double result = -ei(mean, var, best_fitness);
        if(result < min(best, -min_ei)) {
          best_point->assign(x.begin(), x.end());
          best = result;
          has_result = true;
        }
      }
    }
  } else {
    //lambda >= 2
    int num_loops = pow(npts + 1, dimension);
    while(more_viable) {
      vector<double> x(size, 0.0);
      bool can_run = true;
      for(int i = 0; i < lambda; i++) {
        for(int j = 0; j < dimension; j++) {
          x[i * dimension + j] = lower[j] + floor((loop[i] % (int) pow(npts + 1, dimension - j)) / pow(npts + 1, dimension - j - 1)) * steps[j];
          if(x[j] > upper[j % dimension] || x[j] < lower[j % dimension]) can_run = false;
	}
      }
      int i = lambda - 1;
      for(; i >= 0; i--) {
        if(++loop[i] == (num_loops + i - (lambda - 1))) {
          if(i == 0) more_viable = false;
        } else {
          break;
        }
      }
      for(int j = max(i + 1, 1); j < lambda; j++) {
        loop[j] = loop[j-1] + 1;
	if(loop[j] >= num_loops) more_viable = false;
      }

      if(can_run) {
        double result = fitness(x);
        if(result < min(best, -min_ei)) {
          best_point->assign(x.begin(), x.end());
	  best = result;
	  has_result = true;
        }
      }
    }
  }

  if(has_result) {
    if(!suppress) {
      cout << "["; for(int i = 0; i < lambda; i++) {
        for(int j = 0; j < dimension; j++) {
          cout << (*best_point)[i*dimension + j] << " ";
        }
        cout << "\b; ";
      }
      cout << "\b\b] = best = "  << best << endl;
    }
    return best_point;
  } else {
    delete best_point;
    return NULL;
  }
}

vector<double> EGO::brute_search(int npts=10, int lambda=1)
{
  double best = 1000000;
  int size = dimension * lambda;
  vector<double> best_point(size, 0);
  double points[size][npts + 1];
  int num_steps = npts + 1;
  bool has_result = false;

  if(lambda == 1) {
    //Build our steps of possible values in each dimension
    for(int i = 0; i < dimension; i++) {
      if(is_discrete) {
        int step = floor((upper[i] - lower[i]) / npts);
        if(step == 0) step = 1;
        int j = 0;
        for(; j <= npts && (lower[i] + j * step) <= upper[i]; j++) {
          points[i][j] = floor(lower[i] + j * step);
        }
        num_steps = min(num_steps, j);
      } else {
        double step = (upper[i] - lower[i]) / npts;
        int j = 0;
        for(; j <= npts; j++) {
          points[i][j] = lower[i] + j * step;
        }
        num_steps = min(num_steps, j);
      }
    }

    //Loop through each possible combination of values
    for(int i = 0; i < pow(num_steps, dimension); i++) {
      double x[dimension];
      for(int j = 0; j < dimension; j++) {
        x[j] = points[j][((int)(i / pow(num_steps, j))) % num_steps];
      }
      if(not_running(x)) {
        double result = -ei(sg->mean(x), sg->var(x), best_fitness);
        if(result < best) {
          best = result;
          best_point.assign(x, x + dimension);
          has_result = true;
        }
      }
    }
  } else {
    //Build our steps of possible values in each dimension
    for(int i = 0; i < size; i++) {
      if(is_discrete) {
        int step = floor((upper[i % dimension] - lower[i % dimension]) / npts);
        if(step == 0) step = 1;
        int j = 0;
        for(; j <= npts && (lower[i % dimension] + j * step) <= upper[i % dimension]; j++) {
          points[i][j] = floor(lower[i % dimension] + j * step);
        }
        num_steps = min(num_steps, j);
      } else {
        double step = (upper[i % dimension] - lower[i % dimension]) / npts;
        int j = 0;
        for(; j <= npts; j++) {
          points[i][j] = lower[i % dimension] + j * step;
        }
        num_steps = min(num_steps, j);
      }
    }

    //Loop through each possible combination of values
    for(int i = 0; i < pow(num_steps, size); i++) {
      vector<double> x(size, 0.0);
      for(int j = 0; j < size; j++) {
        x[j] = points[j][((int) floor(i / pow(num_steps, j))) % num_steps];
      }
      double result = fitness(x);
      if(result < best) {
        best = result;
        best_point = x;
        has_result = true;
      }
    }
  }

  if(has_result) { 
    return best_point;
  } else {
    if(num_steps > npts || !is_discrete) {
      return brute_search(npts * 2, lambda);
    }
    cout << "Broken, can't brute search" << endl;
    exit(1);
  }
}
