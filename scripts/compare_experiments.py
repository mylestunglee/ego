from plot_experiments import parse_file, read_boxess, max_value
from os.path import basename, normpath
import numpy as np
import sys

if len(sys.argv) != 3:
	print('Usage: python3 compare_kt.py example_old example_new')
	sys.exit(1)

path_old = basename(normpath(sys.argv[1]))
path_new = basename(normpath(sys.argv[2]))
config = parse_file('examples/' + path_new + '/config.txt')

def calc_cost_max(boxes):
	return sum([box[0] for box in boxes])

def calc_score_parameters(boxess):
	fitnesses = [box[1] for boxes in boxess for box in boxes if box[1] < max_value]
	cost_max = max(list(map(calc_cost_max, boxess)))
	return min(fitnesses), max(fitnesses), cost_max

def score_boxes(boxes, fitness_min, fitness_max, cost_max):
	area = boxes[0][0] * fitness_max
	for i in range(len(boxes) - 1):
		area += boxes[i + 1][0] * (min([boxes[i][1], fitness_max]) - fitness_min)
	area += (cost_max - calc_cost_max(boxes)) * (min([boxes[-1][1], fitness_max]) - fitness_min)
	return area / (cost_max * (fitness_max - fitness_min))

# Read boxess
boxess_no_kt = read_boxess('logs/' + path_old + '_' + path_new + '_no_kt', config)
boxess_kt    = read_boxess('logs/' + path_old + '_' + path_new + '_kt',    config)

# Compute scores
fitness_min, fitness_max, cost_max = calc_score_parameters(boxess_no_kt + boxess_kt)

scores_no_kt = [score_boxes(boxes, fitness_min, fitness_max, cost_max) for boxes in boxess_no_kt]
scores_kt    = [score_boxes(boxes, fitness_min, fitness_max, cost_max) for boxes in boxess_kt   ]

# Aggregate scores
scores_no_kt_mean = np.mean(scores_no_kt)
scores_no_kt_sd   = np.std (scores_no_kt)
scores_kt_mean    = np.mean(scores_kt   )
scores_kt_sd      = np.std (scores_kt   )
improvement_mean  = scores_no_kt_mean / scores_kt_mean

ratios = [x / y for x in scores_no_kt for y in scores_kt if y > 0.0]

ratio_mean = np.mean(ratios)
ratio_sd   = np.std (ratios)

improvement_sd = (improvement_mean / ratio_mean) * ratio_sd

print('No KT optimisation score mean: %0.3f' % scores_no_kt_mean)
print('No KT optimisation score sd:   %0.3f' % scores_no_kt_sd  )
print('KT optimisation score mean:    %0.3f' % scores_kt_mean   )
print('KT optimisation score sd:      %0.3f' % scores_kt_sd     )
print('Improvement ratio mean:        %0.3f' % improvement_mean )
print('Improvement ratio sd:         ~%0.3f' % improvement_sd   )
