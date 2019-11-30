import numpy as np 
import math
import sys
from scipy.spatial import distance

# Check if float or not
def isFloat(val):
    try:
        val = float(val)
        return True
    except ValueError:
        return False

# Get data from file and convert to float
def getData(fileName):
    data = []
    with open(fileName) as textFile:
        lines = [line.replace("\n","").split("\t") for line in textFile]
    
    for line in lines:
        data.append([float(f) if isFloat(f) else f for f in line])
    
    # String value handling
    for i in range(len(data[0])):
        if isinstance(data[0][i],str):
            unique_vals = set([line[i] for line in data])
            d = {}
            count = 0
            for val in unique_vals:
                d[val] = count
                count += 1
            for line in data:
                line[i] = d.get(line[i])

    return data

# Split data into training and validation data.
def splitData(index,data,foldSize):
    start = index
    end = index + foldSize
    training_data = data[:start] + data[end:]
    validation_data = data[start:end]
    return training_data,validation_data

# Normalize training data
def normalizeData(min_vals,max_vals,dataset,training_labels):
    # Have to check normalization
    temp = len(dataset[0])
    for i in range(len(dataset)):
        for j in range(temp):
            dataset[i][j] = (dataset[i][j] - min_vals[j])/(max_vals[j]-min_vals[j])
        dataset[i].append(training_labels[i])
    return dataset
    # for i in range(len(dataset)):
    #     normalized = ((dataset[i]-mean)/std_deviation).tolist()
    #     normalized.append(training_labels[i])
    #     dataset[i] = normalized
    # return dataset

# Get the labels of the K-nearest neighbors and then assign label based on maximum occuring label.
def getLabel(nearest_neighbors):
    # Find labels of nearest neighbors and assign label to most commonly occuring label.
    labels = [x[-1] for x in nearest_neighbors]
    # print("Nearest Neigbor labels")
    # print(labels)
    d = dict()
    for i in range(len(labels)):
        if labels[i] in d:
            d[labels[i]] += 1
        else:
            d[labels[i]] = 1
    max_val = max(d.values())
    for key,val in d.items():
        if val==max_val:
            label = key
    # label = [key for key,val in d.items() if val==max_val]
    return label

# Compute the different metrics for evaluation.
def metric_computation(validation_labels,predicted_labels):
    tp,tn,fp,fn = 0.0,0.0,0.0,0.0
    for i in range(len(validation_labels)):
        if(validation_labels[i]==1.0 and predicted_labels[i]==1.0):
            tp += 1
        elif(validation_labels[i]==0.0 and predicted_labels[i]==1.0):
            fp += 1
        elif(validation_labels[i]==0.0 and predicted_labels[i]==0.0):
            tn += 1
        else:
            fn += 1
    
    accuracy = (tp+tn)/(tp+tn+fp+fn) 
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    fmeasure = (2*recall*precision)/(recall+precision)

    return accuracy,precision,recall,fmeasure
        
# Generate the K-nearest neighbors based on Euclidean distance.
def compute_knn(normalized_training_data,test_data,validation_min_vals,validation_max_vals,K):
    dist = []
    nearest_neighbors = []
    length = len(normalized_training_data[0])
    normalized_test_data = []
    for j in range(len(test_data[:-1])):
        val = (test_data[j]-validation_min_vals[j])/(validation_max_vals[j]-validation_min_vals[j])
        normalized_test_data.append(val)
    # normalized_test_data = (test_data[:len(test_data)-1]-mean)/std_deviation
    for i in range(len(normalized_training_data)):
        dist.append((normalized_training_data[i],distance.euclidean(normalized_test_data,normalized_training_data[i][:length-1])))
    dist.sort(key = lambda x:x[1])

    for i in range(K):
        nearest_neighbors.append(dist[i][0])
    label = getLabel(nearest_neighbors)
    return label

# 
def knn_metrics(training_data,validation_data,K):
    dataset = [line[0:-1] for line in training_data]
    training_labels = [line[-1] for line in training_data]
    validation_labels = [line[-1] for line in validation_data]
    predicted_labels = []
    # mean = np.mean(dataset,axis=0)
    # std_deviation = np.std(dataset,axis=0)
    training_min_vals = []
    training_max_vals = []
    dataset_temp = zip(*dataset)
    for row in dataset_temp:
        training_min_vals.append(min(row))
        training_max_vals.append(max(row))
    
    normalized_training_data = normalizeData(training_min_vals,training_max_vals,dataset,training_labels)
    # print("Normalized")
    # print(normalized_training_data)
    validation_min_vals = []
    validation_max_vals = []
    validation_data_temp = zip(*validation_data)[:-1]
    for row in validation_data_temp:
        validation_min_vals.append(min(row))
        validation_max_vals.append(max(row))

    for i in range(len(validation_data)):
        label = compute_knn(normalized_training_data,validation_data[i],validation_min_vals,validation_max_vals,K)
        predicted_labels.append(label)
    
    accuracy,precision,recall,fmeasure = metric_computation(validation_labels,predicted_labels)
    return accuracy,precision,recall,fmeasure

# k-fold cross validation for averaging the results from randomization.
def kFoldCrossValidation(numFolds,data,K):
    foldSize = len(data)/numFolds
    accuracy_list = []
    precision_list = []
    recall_list = []
    fmeasure_list = []
    index = 0
    for i in range(numFolds):
        training_data,validation_data = splitData(index,data,foldSize)
        index += foldSize
        accuracy,precision,recall,fmeasure = knn_metrics(training_data,validation_data,K)
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        fmeasure_list.append(fmeasure)
    
    print("Accuracy: ", np.mean(accuracy_list)*100)
    print("Precision: ", np.mean(precision_list)*100)
    print("Recall: ", np.mean(recall_list)*100)
    print("F-measure:  ",np.mean(fmeasure_list)*100)

# Getting input 
fileName = sys.argv[1]
numOfFolds = int(sys.argv[2])
K = int(sys.argv[3])
data = getData(fileName)
kFoldCrossValidation(numOfFolds,data,K)

