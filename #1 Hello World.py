from sklearn import tree

# Features: [weight, texture]
# Texture: Smooth = 1 , Bumpy = 0
# Labels: Orange = 1, Apple = 0
features = [[150, 0], [170, 0], [140, 1], [130,1]]
labels = [1,1,0,0]
clf = tree.DecisionTreeClassifier()
clf.fit(features, labels)
print clf.predict([[150,0]])