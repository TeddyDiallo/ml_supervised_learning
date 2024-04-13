class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature          # Index of feature to split on
        self.threshold = threshold      # Threshold value for the feature
        self.left = left                # Left child node
        self.right = right              # Right child node
        self.value = value              # Value if the node is a leaf node (class label)

class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth      # Maximum depth of the tree
        self.root = None                # Root node of the decision tree

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        # Check if maximum depth is reached or if all labels are the same
        if depth == self.max_depth or len(set(y)) == 1:
            return Node(value=max(set(y), key=y.count))

        n_samples, n_features = X.shape
        best_gini = float('inf')
        best_feature = None
        best_threshold = None

        # Calculate Gini impurity for each feature and threshold
        for feature in range(n_features):
            thresholds = sorted(set(X[:, feature]))
            for threshold in thresholds:
                left_indices = X[:, feature] <= threshold
                right_indices = X[:, feature] > threshold
                gini = (left_indices.sum() / n_samples) * self._gini_impurity(y[left_indices]) \
                     + (right_indices.sum() / n_samples) * self._gini_impurity(y[right_indices])
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold

        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold
        left_node = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_node = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return Node(feature=best_feature, threshold=best_threshold, left=left_node, right=right_node)

    def _gini_impurity(self, y):
        n_samples = len(y)
        classes = set(y)
        gini = 1
        for c in classes:
            p = (y == c).sum() / n_samples
            gini -= p ** 2
        return gini

    def predict(self, X):
        predictions = []
        for sample in X:
            node = self.root
            while node.left:
                if sample[node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            predictions.append(node.value)
        return predictions
# Instantiate the DecisionTreeClassifier
tree_classifier = DecisionTreeClassifier(max_depth=3)  # You can specify the maximum depth of the tree if needed

# Assuming you have your training data X_train and corresponding labels y_train

# Fit the classifier to the training data
tree_classifier.fit(X_train, y_train)

# Make predictions on the training data
train_predictions = tree_classifier.predict(X_train)

# Optionally, you can evaluate the accuracy on the training data
train_accuracy = accuracy_score(y_train, train_predictions)
print("Training Accuracy:", train_accuracy)
