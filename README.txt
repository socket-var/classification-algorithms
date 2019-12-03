K-nn:

$ python K_Nearest_Neighbors.py project3_dataset1.txt 10 5

$ python K_Nearest_Neighbors_Demo.py project3_dataset3_train.txt project3_dataset3_test.txt 5

Naive Bayes:

$ python3 Naive_Bayes.py project3_dataset1.txt 10
Accuracy:  0.9357142857142857
Precision:  0.9241315440217042
Recall:  0.9053435453435454
F-measure:   0.9132090522262688

$ python3 Naive_Bayes.py project3_dataset2.txt 10
Accuracy:  0.7021739130434783
Precision:  0.5707868713673667
Recall:  0.617493091703618
F-measure:   0.5842003761632244

Decision Tree:

$ python decision_tree.py
Enter the dataset file name: project3_dataset4.txt
Enter number of features to be selected: 
Enter maximum depth of the trees: 
Enter number of folds for validation: 1 
Do you want to print the tree? [y/N]: n

Do you want to test on a test dataset? [y/N]: y
Enter the dataset file name: project3_dataset4.txt

Output:
-> Is feature 0 == overcast? Gain: 0.10204081632653056
--> Predict: 1.0
--> Is feature 2 == high? Gain: 0.18000000000000016
---> Is feature 0 == rain? Gain: 0.11999999999999983
----> Is feature 3 == weak? Gain: 0.5
-----> Predict: 1.0
-----> Predict: 0.0
----> Predict: 0.0
---> Is feature 3 == strong? Gain: 0.11999999999999983
----> Is feature 1 == cool? Gain: 0.5
-----> Predict: 0.0
-----> Predict: 1.0
----> Predict: 1.0

Random Forest:
$ python random_forest.py
Enter the dataset file name: project3_dataset1.txt 
Enter number of trees in the forest: 5
Enter number of features to be randomly selected: 10
Enter maximum depth of the trees: 7
Enter number of folds for validation: 
Enter the size of the bootstrap dataset:
 

Output: 
Accuracy: 0.954323 Precision: 0.945627 Recall: 0.933005 F1 Score: 0.937484