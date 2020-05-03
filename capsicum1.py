import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from IPython.display import Image  
from sklearn import tree
import pydotplus

#data = pd.DataFrame({"seed colour":["Black","Black","Tan","Tan"],
#                     "corolla spots":["False","False","True","True"],
#                     "flowers solitary":["False","False","False","False"],
#                     "species":["pubescens","pubescens","baccatum","annuum"], 
#                     "name":["Rocoto","Canario","Peru yellow","Serrano"]}, 
#                    columns=["seed colour","corolla spots","flowers solitary","species", "name"])

data = pd.DataFrame({"seed colour":["1","1","2","2"],
                     "corolla spots":["0","0","1","1"],
                     "flowers solitary":["0","0","0","0"],
                     "species":["pubescens","pubescens","baccatum","annuum"], 
                     "name":["Rocoto","Canario","Peru yellow","Serrano"]}, 
                    columns=["seed colour","corolla spots","flowers solitary","species", "name"])


#columns=["toothed","hair","breathes","legs","species"])

#features = data[["toothed","hair","breathes","legs"]]

features = data[["seed colour","corolla spots","flowers solitary","species", "name"]]
target = data["species"]
#"Seed colour";"Corolla colour";"Corolla spots";"Flowers solitary";"Filament colour";"Flowers per node";"Name";"Genus";"Species"
print(data)
data1 = data.drop('name', axis=1)
print(data1)


"""
Split the data into a training and a testing set
"""

train_features = data1.iloc[:80,:-1]
test_features = data1.iloc[80:,:-1]
train_targets = data1.iloc[:80,-1]
test_targets = data1.iloc[80:,-1]

"""
Train the model
"""

tree = DecisionTreeClassifier(criterion = 'entropy').fit(train_features,train_targets)

#tree.export_graphviz(clf,
#                     out_file="capsicum.dot",
#                     feature_names = fn, 
#                     class_names=cn,
#                     filled = True)

# Create DOT data
dot_data = tree.export_graphviz(clf, out_file=None, 
                                feature_names=iris.feature_names,  
                                class_names=iris.target_names)

# Draw graph
graph = pydotplus.graph_from_dot_data(dot_data)  

# Show graph
Image(graph.create_png("capsicum1.png"))

"""
Predict the classes of new, unseen data
"""
prediction = tree.predict(test_features)


"""
Check the accuracy
"""

print("The prediction accuracy is: ",tree.score(test_features,test_targets)*100,"%")