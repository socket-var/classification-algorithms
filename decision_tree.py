import helpers
import tree_algos
import numpy as np


if __name__ == "__main__":

    file_name = input("Enter the dataset file name: ")
    n_features = input("Enter number of features to be randomly selected: ")
    max_depth = input("Enter maximum depth of the trees: ")
    k_folds = input("Enter number of folds for validation: ")
    do_print = input("Do you want to print the tree? [y/N]: ")

    if not n_features:
        n_features = None
    else:
        n_features = int(n_features)

    if not max_depth:
        max_depth = None
    else:
        max_depth = int(max_depth)

    if not k_folds:
        k_folds = 10
    else:
        k_folds = int(k_folds)

    # file_name = "project3_dataset2.txt"
    # n_features = None
    # max_depth = None
    # k_folds = 10

    X, y, _ = helpers.import_txt(file_name)

    X = np.array(X)
    y = np.array(y).reshape((-1, 1))

    folds = helpers.cross_validation_split(10, X, y)

    results = []

    for i in range(k_folds):
        X_train = np.array([]).reshape(0, X.shape[1])
        y_train = np.array([]).reshape(0, 1)

        # train on 9 folds
        for idx, fold in enumerate(folds):
            if idx != i:
                [x_fold, y_fold] = fold
                X_train = np.vstack((X_train, x_fold))
                y_train = np.vstack((y_train, y_fold))
        # predict on remaining one fold
        [X_test, y_test] = folds[i]

        classifier = tree_algos.DecisionTreeClassifier(
            n_features=n_features, max_depth=max_depth)

        classifier.fit(X_train, y_train)

        if do_print.lower() == "y":
            print(classifier)

        predictions = classifier.predict(X_test)

        accuracy, precision, recall, fmeasure = helpers.metric_computation(
            y_test, predictions)

        print(accuracy, precision, recall, fmeasure)

        results.append([accuracy, precision, recall, fmeasure])

    measures = np.sum(np.array(results), axis=0) / len(results)
    print("Accuracy: {}, Precision: {}, Recall: {}, F1: {}".format(*measures))

    do_test = input("Do you want to test on a test dataset? [y/N]: ")

    if do_test.lower() == "y":

        classifier = tree_algos.DecisionTreeClassifier(
            n_features=n_features, max_depth=max_depth)

        classifier.fit(X, y)

        test_file_name = input("Enter the test dataset filename: ")

        X_test, y_test, _ = helpers.import_txt(test_file_name)

        predictions = classifier.predict(X_test)

        measures = helpers.metric_computation(
            y_test, predictions)

        print("Accuracy: {}, Precision: {}, Recall: {}, F1: {}".format(*measures))
