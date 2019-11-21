import numpy as np 
import math
import sys

def isFloat(val):
    try:
        val = float(val)
        return True
    except ValueError:
        return False

def getData(fileName):
    data = []
    with open(fileName) as textFile:
        lines = [line.replace("\n","").split("\t") for line in textFile]
    
    for line in lines:
        data.append([float(f) if isFloat(f) else f for f in line])
    
    # Not clear about what to do for string values.
    return data

def splitData(index,data,foldSize):
    start = index
    end = index + foldSize
    training_data = data[:start] + data[end:]
    validation_data = data[start:end]
    return training_data,validation_data

def normalizeData(mean,std_deviation,dataset,training_labels):
    for i in range(len(dataset)):
        normalized = ((dataset[i]-mean)/std_deviation).tolist()
        normalized.append(training_labels[i])
        dataset[i] = normalized
    return dataset

def knn_metrics(training_data,validation_data,K):
    dataset = [line[0:-1] for line in training_data]
    training_labels = [line[-1] for line in training_data]
    validation_labels = [line[-1] for line in validation_data]
    predicted_labels = []

    mean = np.mean(dataset,axis=0)
    std_deviation = np.std(dataset,axis=0)

    normalized_data = normalizeData(mean,std_deviation,dataset,training_labels)

def kFoldCrossValidation(numFolds,data,K):
    foldSize = len(data)/numFolds
    accuracy_list = []
    precision_list = []
    recall_list = []
    fmeasure_list = []
    index = 0
    for i in range(0,numFolds):
        training_data,validation_data = splitData(index,data,foldSize)
        index += foldSize
        accuracy,precision,recall,fmeasure = knn_metrics(training_data,validation_data,K)

# Getting input 
fileName = sys.argv[1]
numOfFolds = int(sys.argv[2])
K = int(sys.argv[3])
data = getData(fileName)
kFoldCrossValidation(numOfFolds,data,K)

