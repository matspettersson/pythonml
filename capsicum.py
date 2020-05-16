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
#import pickle
from joblib import dump, load

# git add -u
# git commit -m ""
# git push origin master
# https://machinelearningmastery.com/machine-learning-in-python-step-by-step/

filename = 'capsicum.csv'

def capsicum(filename):
   df = pd.read_csv(filename, sep=';')
   print('*** df ***')
   print(df.describe())
   print(df.head(25))

   #dd1 = pd.get_dummies(df[['seed colour','corolla colour','corolla spots','flowers solitary','flowers per node','species' ] ] )
   dd1 = pd.get_dummies(df[['seed colour','corolla colour','corolla spots','flowers solitary','flowers per node'] ] )
   print('*** dd1 describe ***')
   print(dd1.describe())
   print('*** dd1 head ***')
   print(dd1.head(25))

   
   for col in dd1.columns: 
      print(col) 
   clf = DecisionTreeClassifier(criterion='gini',max_depth = 8, random_state=0)
   #clf = DecisionTreeClassifier()
   #clf_train = clf.fit(df1, df['seed colour']
   # )
   #clf_train = clf.fit(df1, df1['species'])
   clf_train = clf.fit(dd1, df['species'])

   #X_train, X_test, Y_train, Y_test = train_test_split(df2, df2, random_state=10)
   # Step 3: Train the model on the data
   #clf.fit(X_train, Y_train)
   
   #y_pred = clf_train.predict(X_test)
   #print(y_pred)
   # Export/Print a decision tree in DOT format.
   dot_data = StringIO()
   tree.export_graphviz(clf_train, out_file=dot_data,feature_names=list(dd1.columns.values),  rounded=True, filled=True)
   #class_names=['species', 'name'],
   graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
   graph.write_png("capsicum.png")

   #s = pickle.dumps(clf_train)
   dump(clf_train, 'capsicum.joblib') 
   #clf2 = pickle.loads(s)
   #clf2.predict(X[0:1])


def main():
   capsicum(filename)

if __name__ == "__main__":
    main()
