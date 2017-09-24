from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import csv
import sys

if len(sys.argv) != 4:
	print('Usage: plot_results.py config results output')
	sys.exit(1)

# Read CSV file
config_file = open(sys.argv[1], 'r')
config_reader = csv.reader(config_file, delimiter = ',')

settings = []
for row in config_reader:
	if len(row) != 0 and row[0][0] != '#':
		settings.append(row)

if (len(settings[6]) != 2):
	print('Cannot plot non-two-dimensional functions')
	sys.exit(1)

results_file = open(sys.argv[2], 'r')
results_reader = csv.reader(results_file, delimiter = ',')

xs = []
ys = []
zs = []
for row in results_reader:
	xs.append(float(row[0]))
	ys.append(float(row[1]))
	zs.append(float(row[2]))

# Plot

# Use Latex font
plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')

fig = plt.figure()

ax = fig.gca(projection = '3d')
ax.plot_trisurf(xs, ys, zs, cmap = cm.plasma, linewidth = 0.0)

def format_label(text):
	return text.replace('_', ' ').capitalize()

ax.set_xlabel(format_label(settings[10][0]))
ax.set_ylabel(format_label(settings[10][1]))
ax.set_zlabel('Benchmark execution time (s)')

# Normalise axes
def get_interval(xs):
	return (max(xs) - min(xs)) / (len(xs) - 1.0)

if get_interval(list(ax.get_xticks())) <= 1.0:
	ax.set_xticks(np.arange(min(xs), max(xs)+1, 1.0))

if get_interval(list(ax.get_yticks())) <= 1.0:
	ax.set_yticks(np.arange(min(ys), max(ys)+1, 1.0))

# Rotation
ax.view_init(azim=135)

plt.savefig(sys.argv[3])

