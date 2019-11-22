import sys
import csv

## Checks if given string is a number
def isNum(s):
	try:
		float(s)
		return True
	except:
		return False

## Normalizes data
def normalizeData(data,norm_min,norm_max,columnLabels):
	normalizedData = []
	for row in data:
		normRow = []
		for i in range(0,len(row)):
			if(columnLabels[i] == "Numerical"):
				normRow.append((row[i]-norm_min[i])/(norm_max[i]-norm_min[i]))
			else:
				normRow.append(row[i])
		normalizedData.append(normRow)
	return normalizedData


filename = sys.argv[1]
data = []
rowCount = 0

## Reading file and storing it as a matrix
with open(filename) as csv_file:
	csv_reader = csv.reader(csv_file, delimiter='\t')
	for row in csv_reader:
		data.append([float(f) if isNum(f) else f for f in row])

columnLabels = []
sampleData = data[0]

## Getting column labels
for i in range(0,len(sampleData)-1):
	if(type(sampleData[i]) == str):
		columnLabels.append("Categorical")
	else:
		columnLabels.append("Numerical")
columnLabels.append("Class")

norm_min = []
norm_max = []

for i in range(0,len(columnLabels)):
	if(columnLabels[i] == "Numerical"):
		temp = [row[i] for row in data]
		norm_min.append(min(temp))
		norm_max.append(max(temp))
	else:
		norm_min.append("Categorical")
		norm_max.append("Categorical")

normalizedData = normalizeData(data,norm_min,norm_max,columnLabels)




		