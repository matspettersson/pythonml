import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
#from sklearn.datasets import load_iris
from sklearn import tree
import pandas as pd
import numpy as np
from sklearn import tree
from IPython.display import Image  
import pydotplus
from io import StringIO


# git add -u
# git commit -m ""
# git push origin master
# https://machinelearningmastery.com/machine-learning-in-python-step-by-step/

filename = 'capsicum.csv'

df = pd.read_csv(filename, sep=';')
print(df.describe())
print(df.head())

# Step 2: Make an instance of the Model
#clf = DecisionTreeClassifier(max_depth = 2, random_state = 0)
clf = DecisionTreeClassifier()
df1 = pd.get_dummies(df['seed colour'],['species'])
clf_train = clf.fit(df1, df1)

X_train, X_test, Y_train, Y_test = train_test_split(df1, df1, random_state=0)

# Step 3: Train the model on the data
#clf.fit(X_train, Y_train)
#clf.fit(df['Seed colour']['Filament colour'], df['Corolla spots']['Flowers per node'])
# Step 4: Predict labels of unseen (test) data


print("DF")
print(df.describe())
print(df.head())
print("DF1")
print(df1.describe())
print(df1.head())
print("X_train")
print(X_train.describe())
print(X_train.head())

print("Y_train")
print(Y_train.describe())
print(Y_train.head())

clf_train = clf.fit(X_train, Y_train)
# Export/Print a decision tree in DOT format.
dot_data = StringIO()
tree.export_graphviz(clf_train, out_file=dot_data,feature_names=list(df1.columns.values))
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png("capsicum.png")

#dot_data = tree.export_graphviz(clf_train, out_file=None, feature_names=list(one_hot_data.columns.values), 
#                                class_names=['Not_Play', 'Play'], rounded=True, filled=True) #Gini decides which attribute/feature should be placed at the root node, which features will act as internal nodes or leaf nodes#Create Graph from DOT data
