import helpers
import numpy as np
import pandas as pd


class Question:

    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, row):

        val = row[self.column]

        if isinstance(val, float):
            return val >= self.value

        elif isinstance(val, str) or self.column == len(row)-1:
            return val == self.value


class DecisionTreeClassifier:

    def fit(self, X, y):

        self.data = np.concatenate((X, y), axis=1)
        self.tree = self._make_tree(self.data)

    def predict(self, X, y):

        predictions = np.empty(y.shape)

        self.data = np.concatenate((X, y), axis=1)

        for idx, row in enumerate(self.data):

            label = self._get_leaf_prediction(self._classify(row, self.tree))

            predictions[idx] = label

        return predictions

    def _make_tree(self, subset, level=0):

        best_gain, best_question = self._best_split(subset)

        if best_gain == 0:
            return self._make_leaf_node(subset)

        true_set, false_set = self._partition(subset, best_question)

        true_branch = self._make_tree(true_set, level+1)

        false_branch = self._make_tree(false_set, level+1)

        return self._make_decision_node(best_question, true_branch, false_branch)

    def _best_split(self, subset):
        best_gain = 0
        best_question = None
        selected_cols = set()

        curr_gini = helpers.gini(subset)
        num_cols = len(subset[0]) - 1

        for col in range(num_cols):

            if col not in selected_cols:
                values = set([row[col] for row in subset])

                for val in values:

                    curr_question = Question(col, val)

                    true_subset, false_subset = self._partition(
                        subset, curr_question)

                    if len(true_subset) == 0 or len(false_subset) == 0:
                        continue

                    curr_gain = helpers.info_gain(
                        true_subset, false_subset, curr_gini)

                    if curr_gain >= best_gain:
                        best_gain, best_question = curr_gain, curr_question
                        selected_cols.add(col)

        return best_gain, best_question

    def _get_leaf_prediction(self, counts):

        max_label = None
        _max = -1
        total = sum(counts.values())

        for label in counts.keys():
            certainity = float(counts[label]) / total

            if certainity > _max:
                _max = certainity
                max_label = label

        return float(max_label)

    def _partition(self, subset, question):

        true_subset, false_subset = [], []
        for row in subset:
            if question.match(row):
                true_subset.append(row)
            else:
                false_subset.append(row)
        return true_subset, false_subset

    def _classify(self, row, node):

        if node["type"] == "leaf":
            return node["predictions"]
        else:
            if node["question"].match(row):
                return self._classify(row, node["true_branch"])
            else:
                return self._classify(row, node["false_branch"])

    def _make_leaf_node(self, subset):

        unique, counts = np.unique(np.array(subset)[:, -1], return_counts=True)
        counts = dict(zip(unique, counts))

        return {
            "type": "leaf",
            "predictions": counts
        }

    def _make_decision_node(self, question, true_branch, false_branch):

        return {
            "type": "decision",
            "question": question,
            "true_branch": true_branch,
            "false_branch": false_branch
        }


if __name__ == "__main__":

    # file_name = input("Enter the dataset file name: ")
    file_name = "project3_dataset1.txt"

    X, y, _ = helpers.import_txt(file_name)

    X = np.array(X)
    y = np.array(y).reshape((-1, 1))

    folds = helpers.cross_validation_split(10, X, y)

    for i in range(10):
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

        classifier = DecisionTreeClassifier()

        classifier.fit(X_train, y_train)

        predictions = classifier.predict(X_test, y_test)

        accuracy, precision, recall, fmeasure = helpers.metric_computation(
            y_test, predictions)

        print(accuracy, precision, recall, fmeasure)
