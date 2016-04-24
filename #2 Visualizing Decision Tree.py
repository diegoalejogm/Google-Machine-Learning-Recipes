import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()
test_idx = [0,50,100]

# Training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# Testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

# Train classifier
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)


print test_target
print clf.predict(test_data)

from sklearn.externals.six import StringIO
import pydot
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data, feature_names=iris.feature_names, class_names = iris.target_names, filled = True, rounded= True, impurity=False)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")

# Visualize labels: print iris.target
# Visualize Feature Names: print iris.feature_names
# Visualize Feature Names: print iris.target_names
# Visualize First data: print iris.data[0]