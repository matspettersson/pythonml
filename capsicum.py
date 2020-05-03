import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
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

df = pd.read_csv('capsicum.csv', sep=';')
print(df.describe())
print(df.head())

#X_train, X_test, Y_train, Y_test = train_test_split(df[data.feature_names], df['target'], random_state=0)
#X_train, X_test, Y_train, Y_test = train_test_split(df, df, random_state=0)

# Step 2: Make an instance of the Model
#clf = DecisionTreeClassifier(max_depth = 2, random_state = 0)
# Step 3: Train the model on the data
#clf.fit(X_train, Y_train)
#clf.fit(df['Seed colour']['Filament colour'], df['Corolla spots']['Flowers per node'])
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
#pd1 = pd.get_dummies(pd[ ['seed colour', 'corolla spots', 'flowers solitary', 'species'] ])
df1 = pd.get_dummies(df[ ['seed colour', 'species'] ])

#columns=["seed colour","corolla spots","flowers solitary","species", "name"])
print(df1)

# The decision tree classifier.
clf = tree.DecisionTreeClassifier()
clf_train = clf.fit(df1, df['species'])

# Export/Print a decision tree in DOT format.
#print(tree.export_graphviz(clf_train, None))

#Create Dot Data
#dot_data = tree.export_graphviz(clf_train, out_file=None, feature_names=list(one_hot_data.columns.values), 
#                                class_names=['Not_Play', 'Play'], rounded=True, filled=True) #Gini decides which attribute/feature should be placed at the root node, which features will act as internal nodes or leaf nodes#Create Graph from DOT data
#graph = pydotplus.graph_from_dot_data(dot_data)

# Show graph
#Image(graph.create_png("png"))


dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data,feature_names=list(df1.columns.values))
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("capsicum.pdf")
graph.write_png("capsicum.png")

#dot_data = tree.export_graphviz(clf_train, out_file=None, feature_names=list(one_hot_data.columns.values), 
#                                class_names=['Not_Play', 'Play'], rounded=True, filled=True) #Gini decides which attribute/feature should be placed at the root node, which features will act as internal nodes or leaf nodes#Create Graph from DOT data
