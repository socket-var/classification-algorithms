import numpy as np
from scipy.stats import mode
import helpers


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

    def __repr__(self):
        condition = "=="

        if isinstance(self.value, float):
            condition = ">="

        return "Is feature %s %s %s?" % (self.column, condition, str(self.value))


class DecisionTreeClassifier:

    def __init__(self, max_depth=None, n_features=None):
        self.tree = None
        self.n_features = n_features
        self.max_depth = max_depth

    def fit(self, X, y):

        self.data = np.concatenate((X, y), axis=1)
        self.tree = self._make_tree(self.data)

    def predict(self, X):

        predictions = np.empty((X.shape[0], 1))

        for idx, row in enumerate(X):

            label = self._classify(row, self.tree)

            predictions[idx] = label

        return predictions.flatten()

    def _make_tree(self, subset, level=0):

        if self.max_depth and level == self.max_depth:
            return

        best_gain, best_question, best_gini = self._best_split(subset)

        if best_gain == 0:
            return self._make_leaf_node(best_gini, best_gain, subset)

        true_set, false_set = self._partition(subset, best_question)

        true_branch = self._make_tree(true_set, level+1)

        false_branch = self._make_tree(false_set, level+1)

        return self._make_decision_node(best_gini, best_gain, best_question, true_branch, false_branch)

    def _best_split(self, subset):

        best_gain = best_gini = 0
        best_question = None
        selected_cols = set()

        curr_gini = helpers.gini(subset)
        num_cols = len(subset[0]) - 1

        columns = list(range(num_cols))

        uniq_cols = set()

        if self.n_features:
            while len(uniq_cols) < self.n_features:
                sample_col = np.random.choice(num_cols, 1)[0]
                if sample_col not in uniq_cols:
                    uniq_cols.add(sample_col)

            columns = list(uniq_cols)

        for col in columns:

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
                        best_gain, best_question, best_gini = curr_gain, curr_question, curr_gini
                        selected_cols.add(col)

        return best_gain, best_question, best_gini

    def _partition(self, subset, question):

        true_subset, false_subset = [], []
        for row in subset:
            if question.match(row):
                true_subset.append(row)
            else:
                false_subset.append(row)
        return true_subset, false_subset

    def _classify(self, row, node):
        # if max depth doesnt produce convergence
        if not node:
            return np.random.choice(np.unique(self.data[:, -1]))
        elif node["type"] == "leaf":
            return node["prediction"]
        else:
            if node["question"].match(row):
                return self._classify(row, node["true_branch"])
            else:
                return self._classify(row, node["false_branch"])

    def _make_leaf_node(self, gini, gain, subset):

        unique, counts = np.unique(np.array(subset)[:, -1], return_counts=True)

        m = counts.argmax()

        return {
            "type": "leaf",
            "prediction": float(unique[m]),
            "gini": gini,
            "gain": gain
        }

    def _make_decision_node(self, gini, gain, question, true_branch, false_branch):

        return {
            "type": "decision",
            "question": question,
            "true_branch": true_branch,
            "false_branch": false_branch,
            "gini": gini,
            "gain": gain
        }

    def __repr__(self):

        tree_repr = []

        self._print_tree_recursive(self.tree, tree_repr)

        return "\n".join(tree_repr)

    def _print_tree_recursive(self, node, tree_repr, depth=1):

        if isinstance(node, dict):

            if node["type"] == "leaf":
                tree_repr.append("{}> Predict: {}".format(
                    depth*'-', node["prediction"]))
                return

            tree_repr.append("{}> {} Gini: {} Gain: {}".format(
                depth*'-', node["question"], node["gini"], node["gain"]))

            self._print_tree_recursive(node['true_branch'], tree_repr, depth+1)
            self._print_tree_recursive(
                node['false_branch'], tree_repr, depth+1)


class RandomForestClassifier:

    def __init__(self, n_estimators=1, n_features=None, max_depth=None):
        self.n_estimators = n_estimators
        self.n_features = n_features
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):

        for _ in range(self.n_estimators):

            # TODO: don't hardcode bootstrap size
            data = np.concatenate((X, y), axis=1)
            data_bootstrapped = self._get_bootstrap_set(data, data.shape[0])

            X_bootstrapped, y_bootstrapped = data_bootstrapped[:,
                                                               :-1], data_bootstrapped[:, -1].reshape((-1, 1))

            tree = DecisionTreeClassifier(
                n_features=self.n_features, max_depth=self.max_depth)

            tree.fit(X_bootstrapped, y_bootstrapped)

            self.trees.append(tree)

    def predict(self, X):
        predictions = []

        for tree in self.trees:
            pred = tree.predict(X).reshape((-1, 1))

            predictions.append(pred)

        return np.array(mode(np.array(predictions), axis=0)).flatten()

    def _get_bootstrap_set(self, data, dataset_length):

        indices = np.random.randint(0, data.shape[0], dataset_length)

        subset = data[indices].copy()

        return subset

    def __repr__(self):

        tree_repr = []

        for idx, tree in enumerate(self.trees):
            tree_repr.append("Tree {}".format(idx+1))

            tree_repr.append(str(tree))

        return "\n".join(tree_repr)
