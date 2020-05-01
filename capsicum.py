import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import tree

# git add -u
# git commit -m ""
# git push origin master
# https://machinelearningmastery.com/machine-learning-in-python-step-by-step/

df = pd.read_csv('capsicum.csv', sep=';')
print(df.describe())
print(df.head())

#df = pd.DataFrame(data.data, columns=data.feature_names)
#df['target'] = data.target
#X_train, X_test, Y_train, Y_test = train_test_split(df[data.feature_names], df['target'], random_state=0)
X_train, X_test, Y_train, Y_test = train_test_split(df, df, random_state=0)

# Step 2: Make an instance of the Model
clf = DecisionTreeClassifier(max_depth = 2, random_state = 0)
# Step 3: Train the model on the data
clf.fit(X_train, Y_train)
# Step 4: Predict labels of unseen (test) data
# Not doing this step in the tutorial
# clf.predict(X_test)


#fn=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
#cn=['setosa', 'versicolor', 'virginica']
#fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
#tree.plot_tree(clf, feature_names = fn, class_names=cn, filled = True);
#fig.savefig('capsicum.png')
#tree.export_graphviz(clf, out_file="capsicum.dot", feature_names = fn, class_names=cn, filled = True)
#tree.plot_tree(clf);
