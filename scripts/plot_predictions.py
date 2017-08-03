from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import csv
import sys

if len(sys.argv) != 3:
	print('Usage: plot_predictions.py predictions.csv output')
	sys.exit(1)

# Read CSV file
csvfile = open(sys.argv[1], 'r')
spamreader = csv.reader(csvfile, delimiter = ',')

xs = []
ys = []
zs = []
for row in spamreader:
	xs.append(float(row[0]))
	ys.append(float(row[1]))
	zs.append(float(row[-2]))

#X, Y = np.meshgrid(xs, ys)
#temp = np.array(zs)
#Z = temp.reshape(X.shape)

# Plot
fig = plt.figure()

ax = fig.gca(projection = '3d')
ax.plot_trisurf(xs, ys, zs, cmap = cm.magma, linewidth = 0.0)
plt.savefig(sys.argv[2])

