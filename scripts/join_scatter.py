import matplotlib.pyplot as plt
import csv
import sys

if len(sys.argv) < 5:
	print('Usage: join_scatter.py results.csv results.csv slice output')
	exit(1)

csvfile1 = open(sys.argv[1], 'r')
csvfile2 = open(sys.argv[2], 'r')
slice = int(sys.argv[3])
spamreader1 = csv.reader(csvfile1, delimiter = ',')
spamreader2 = csv.reader(csvfile2, delimiter = ',')

data = {}

for row in spamreader1:
#	s = row[0]
#	if int(s) != slice:
#		continue
	xs = tuple(row[:len(row) - 2])
	y = float(row[-1])
	data[xs] = (y, None)

for row in spamreader2:
#	s = row[0]
#	if int(s) != slice:
#		continue
	xs = tuple(row[:len(row) - 2])
	z = float(row[-1])
	if xs in data:
		(y, _) = data[xs]
		data[xs] = (y, z)

ys = []
zs = []
for (y, z) in data.values():
	if z == None:
		continue
	ys.append(y)
	zs.append(z)

plt.plot(ys, zs, '.')
plt.xlabel('Robot fitness')
plt.ylabel('Stochastic fitness')
plt.title('Correlation graph')
plt.savefig(sys.argv[4])
