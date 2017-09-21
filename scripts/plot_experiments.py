from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import csv
import sys
from os import listdir
from os.path import isfile, join, basename, normpath

if len(sys.argv) < 3 or len(sys.argv) > 5:
	print('Usage: plot_experiments example_old example_new [output] [max_cost]')
	sys.exit(1)

# Get designs
path_old = basename(normpath(sys.argv[1]))
path_new = basename(normpath(sys.argv[2]))
# Upper bound for fitness values
max_value = 10.0 ** 308.0

# Read cost
max_cost = max_value
if len(sys.argv) == 5:
	max_cost = float(sys.argv[4])

def parse_file(file):
	csvfile = open(file, 'r')
	spamreader = csv.reader(csvfile, delimiter = ',')
	rows = []
	for row in spamreader:
		if len(row) > 0 and row[0][0] != '#':
			rows.append(row)
	return rows

def parse_results(rows, config):
	# Read configuration file
	dimension = len(config[6])
	constraints = int(config[4][0])
	count_costs = int(config[5][0])
	boxes = []
	# Extract fitness cost pair for each row
	for row in rows:
		values = list(map(float, row))
		fitness = values[dimension]
		label = values[dimension + 1]
		cost = 1.0
		if count_costs > 0:
			cost = sum(values[dimension + 2 + constraints :])
		if label != 0.0:
			fitness = max_value

		boxes.append((cost, fitness))
	return boxes

def aggregate_widths(boxess):
	np.cumsum

def aggregate_widths(boxess):
	# Get list of list of costs
	costss = [[box[0] for box in boxes] for boxes in boxess]
	widths = []
	while len(costss) > 0:
		heads = [costs[0] for costs in costss]
		width = min(heads)
		widths.append(width)

		def pop_width(costs):
			head = costs[0]
			if width == head:
				return costs[1:]
			if width < head:
				return [head - width] + costs[1:]
			return costs

		# Remove empty costs
		costss = list(filter(None, map(pop_width, costss)))
	return widths

def partition_by_widths(boxess_old, widths):
	boxess_new = []
	for width in widths:
		if len(boxess_old) == 0:
			boxess_new.append((width, max_value))
		elif width < boxess_old[0][0]:
			boxess_old[0] = (boxess_old[0][0] - width, boxess_old[0][1])
			boxess_new.append((width, boxess_old[0][1]))
		elif width == boxess_old[0][0]:
			boxess_new.append(boxess_old.pop(0))
		else:
			print('Invalid width')
	return boxess_new

def decrease_boxes(boxes_old):
	boxes_new = []
	height = max_value
	for i in range(len(boxes_old)):
		width = boxes_old[i][0]
		height = min(boxes_old[i][1], height)
		boxes_new.append((width, height))
	return boxes_new

def aggregate_boxess(boxess):
	means = []
	sds = []
	accum = []
	sum = 0.0
	for i in range(len(boxess[0])):
		cost = boxess[0][i][0]
		fitnesses = [boxes[i][1] for boxes in boxess if boxes[i][1] < max_value]
		sum += cost
		if len(fitnesses) == 0:
			continue
		means.append(np.mean(fitnesses))
		sds.append(np.std(fitnesses))
		accum.append(sum)
		print(i)
	return means, sds, accum

def read_boxess(directory, config):
	print('Scanning directory')
	files = [join(directory, f) for f in listdir(directory) if isfile(join(directory, f))]
	print('Reading files')
	rowss = [parse_file(file) for file in files]
	print('Parsing results')
	boxess = [parse_results(rows, config) for rows in rowss]
	print('Aggregating widths')
	widths = aggregate_widths(boxess)
	print('Paritioning widths')
	boxess = [partition_by_widths(boxes, widths) for boxes in boxess]
	print('Decreasing boxes')
	boxess = list(map(decrease_boxes, boxess))
	return boxess

def parse_directory(directory, config, label, color):
	boxess = read_boxess(directory, config)
	print('Aggregating widths')
	widths = [box[0] for box in boxess[0]]
	print('Aggregating boxes')
	means, sds, accum = aggregate_boxess(boxess)

	nexts = accum[1:]

	# Plot horizontal lines
	plt.hlines(means[:-1], accum[:-1], nexts, color = color, label = label)

	# Plot confidence interval
	for i in range(len(means) - 1):
		plt.fill_between([accum[i], nexts[i]], [means[i] - sds[i]] * 2, [means[i] + sds[i]] * 2, alpha = 0.25, edgecolor = 'none', facecolor = color)

# Do not plot without output
if len(sys.argv) != 3:
	config = parse_file('examples/' + path_new + '/config.txt')

	# Use Latex font
	plt.rc('text', usetex = True)
	plt.rc('font', family = 'serif')

	boxess = parse_directory('logs/' + path_old + '_' + path_new + '_no_kt', config, 'No KT', 'r')
	boxess = parse_directory('logs/' + path_old + '_' + path_new + '_kt', config, 'KT', 'b')

	lims = plt.xlim()
	plt.xlim(lims[0], min([lims[1], max_cost]))

	plt.legend()
	plt.xlabel('Optimisation time (s)')
	plt.ylabel('Benchmark execution time (s)')
	plt.savefig(sys.argv[3])

