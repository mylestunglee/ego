import csv
import sys

if len(sys.argv) < 2:
	print('Usage: python3 lookup.py results.csv [x_1, ...]')
	sys.exit(1)

file = open(sys.argv[1], 'r')
spamreader = csv.reader(file)

x = sys.argv[2:]
dimension = len(x)

for row in spamreader:
	if list(map(int, row[: dimension])) == list(map(float, x)):
		for a in row[dimension :]:
			print(a)
		break
