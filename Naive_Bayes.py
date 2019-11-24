import sys
import csv
import math
import scipy.stats as s

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
			posteriorPositiveStd.append(math.sqrt(positiveSum/positiveCount))
			posteriorNegativeStd.append(math.sqrt(negativeSum/negativeCount))
		else:
			posteriorPositiveStd.append("NaN")
			posteriorNegativeStd.append("NaN")
	return posteriorPositiveStd,posteriorNegativeStd

# k-fold cross validation for averaging the results from randomization.
def kFoldCrossValidation(numFolds,data,columnLabels):
	foldSize = int(len(data)/numFolds)
	accuracy_list = []
	precision_list = []
	recall_list = []
	fmeasure_list = []
	index = 0
	for i in range(numFolds):
		training_data,validation_data = splitData(index,data,foldSize)
		index += foldSize
		norm_max = []
		norm_min = []
		for i in range(0,len(columnLabels)):
			if(columnLabels[i] == "Numerical"):
				temp = [row[i] for row in training_data]
				norm_min.append(min(temp))
				norm_max.append(max(temp))
			else:
				norm_min.append("Categorical")
				norm_max.append("Categorical")
		#normalizedTrainingData = normalizeData(training_data,norm_min,norm_max,columnLabels)
		#normalizedValidationData = normalizeData(validation_data,norm_min,norm_max,columnLabels)
		normalizedTrainingData = training_data
		normalizedValidationData = validation_data
		positivePrior,negativePrior,classColumn = calculatePriorProbability(normalizedTrainingData)
		posteriorPositiveMean,posteriorNegativeMean = calculatePosteriorMean(normalizedTrainingData,columnLabels,classColumn)
		posteriorPositiveStd,posteriorNegativeStd = calculatePosteriorStandardDeviation(normalizedTrainingData,posteriorPositiveMean,posteriorNegativeMean,columnLabels,classColumn)
		prediction,groundTruth = predictClass(normalizedTrainingData,normalizedValidationData,positivePrior,negativePrior,posteriorPositiveMean,posteriorNegativeMean,posteriorPositiveStd,posteriorNegativeStd,columnLabels,classColumn)
		print(prediction)

# Predicts the class depending on probability
def predictClass(Tdata,Vdata,pPrior,nPrior,pMean,nMean,pStd,nStd,columnLabels,classColumn):
	groundTruth = []
	print(Tdata[0])
	for i in range(0,len(data)):
		groundTruth.append(Tdata[i][-1])
		query = Tdata[i][0:-1]
		positiveClassProbability = calculateClassConditionalProbability(query,Tdata,pPrior,pMean,pStd,columnLabels,classColumn,1)
		negativeClassProbability = calculateClassConditionalProbability(query,Tdata,nPrior,nMean,nStd,columnLabels,classColumn,0)


def calculateClassConditionalProbability(query,Tdata,prior,postMean,postStd,columnLabels,classColumn,classFlag):
	posteriorProbabilities = []
	for i in range(0,len(query)):
		if(columnLabels[i] == "Numerical"):
			posteriorProbabilities.append(calculateNumericalProbability(query[i],postMean[i],postStd[i]))
		else:
			trainColumn = []
			for j in range(0,len(Tdata)):
				trainColumn.append(Tdata[j][i])
			posteriorProbabilities.append(calculateCategoricalProbability(query[i],trainColumn,classColumn,classFlag))
	print(posteriorProbabilities)
	raise NotImplementedError

def calculateNumericalProbability(columnVal,mean,std):
	#print(columnVal,mean,std)
	# a = 1/(math.sqrt(2)*math.sqrt(math.pi)*std)
	# b = math.exp(-1*(math.pow((columnVal-mean),2)/(2*math.pow(std,2))))
	# print(a,b)
	# return a*b
	return s.norm(mean,std).pdf(columnVal)

def calculateCategoricalProbability(columnVal,trainColumn,classColumn,catFlag):
	count = 0
	totalCount = 0
	for i in range(0,len(classColumn)):
		if(classColumn[i] == catFlag):
			totalCount+=1
	for i in range(0,len(trainColumn)):
		if(classColumn[i] == catFlag and columnVal == trainColumn[i]):
			count+=1
	return count/totalCount

# Split data into training and validation data.
def splitData(index,data,foldSize):
	start = index
	end = index + foldSize
	training_data = data[:start] + data[end:]
	validation_data = data[start:end]
	return training_data,validation_data

filename = sys.argv[1]
numFolds = int(sys.argv[2])
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

kFoldCrossValidation(numFolds,data,columnLabels)

# normalizedData = normalizeData(data,norm_min,norm_max,columnLabels)
# positivePrior,negativePrior,classColumn = calculatePriorProbability(normalizedData)
# posteriorPositiveMean,posteriorNegativeMean = calculatePosteriorMean(normalizedData,columnLabels,classColumn)
# posteriorPositiveStd,posteriorNegativeStd = calculatePosteriorStandardDeviation(normalizedData,posteriorPositiveMean,posteriorNegativeMean,columnLabels,classColumn)







		