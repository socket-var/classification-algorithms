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

## Get class probalbilities
def calculatePriorProbability(normalizedData):
	classColumn = []
	positveCount = 0
	negativeCount = 0
	classIndex = len(normalizedData[0])-1
	for row in normalizedData:
		classColumn.append(row[classIndex])
	for val in classColumn:
		if(val == 1):
			positveCount+=1
		else:
			negativeCount+=1
	return positveCount/len(normalizedData),negativeCount/len(normalizedData),classColumn

## Get Posterior Mean for numerical columns
def calculatePosteriorMean(normalizedData,columnLabels,classColumn):
	posteriorPositiveMean = []
	posteriorNegativeMean = []
	positiveCount = 0
	negativeCount = 0
	for val in classColumn:
		if val == 1:
			positiveCount+=1
		else:
			negativeCount+=1
	for i in range(0,len(normalizedData[0])-1):
		if(columnLabels[i] == "Numerical"):
			positiveSum=0
			negativeSum=0
			for j in range(0,len(normalizedData)):
				if(classColumn[j] == 1):
					positiveSum+=normalizedData[j][i]
				else:
					negativeSum+=normalizedData[j][i]
			posteriorPositiveMean.append(positiveSum/positiveCount)
			posteriorNegativeMean.append(negativeSum/negativeCount)
		else:
			posteriorPositiveMean.append("NaN")
			posteriorNegativeMean.append("NaN")
	return posteriorPositiveMean,posteriorNegativeMean

## Get Posterior Standard deviation for numerical columns
def calculatePosteriorStandardDeviation(normalizedData,posteriorPositiveMean,posteriorNegativeMean,columnLabels,classColumn):
	posteriorPositiveStd = []
	posteriorNegativeStd = []
	positiveCount = 0
	negativeCount = 0
	for val in classColumn:
		if val == 1:
			positiveCount+=1
		else:
			negativeCount+=1
	positiveCount = positiveCount-1
	negativeCount = negativeCount-1
	for i in range(0,len(normalizedData[0])-1):
		if(columnLabels[i] == "Numerical"):
			positiveSum = 0
			negativeSum = 0
			for j in range(0,len(normalizedData)):
				if(classColumn[j] == 1):
					positiveSum+= (normalizedData[j][i] - posteriorPositiveMean[i])**2
				else:
					negativeSum+= (normalizedData[j][i] - posteriorNegativeMean[i])**2
			posteriorPositiveStd.append(positiveSum/positiveCount)
			posteriorNegativeStd.append(negativeSum/negativeCount)
		else:
			posteriorPositiveStd.append("NaN")
			posteriorNegativeStd.append("NaN")
	return posteriorPositiveStd,posteriorNegativeStd


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
positivePrior,negativePrior,classColumn = calculatePriorProbability(normalizedData)
posteriorPositiveMean,posteriorNegativeMean = calculatePosteriorMean(normalizedData,columnLabels,classColumn)
posteriorPositiveStd,posteriorNegativeStd = calculatePosteriorStandardDeviation(normalizedData,posteriorPositiveMean,posteriorNegativeMean,columnLabels,classColumn)
print(posteriorPositiveStd)
print(posteriorNegativeStd)






		