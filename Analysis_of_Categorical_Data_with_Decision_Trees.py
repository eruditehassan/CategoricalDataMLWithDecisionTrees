#Importing Required Modules
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import tree
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import graphviz

# Reading Data

df = pd.read_csv("2_analcatdata_broadway.csv", sep=";")

# Separating and Splitting
X = df.loc[:,df.columns!="label"]
y = df["label"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

#Building the Classifier, calculating Accuracy and Plotting using Matplotlib

# - Used sklearn to build a Decision Tree Classifier
# - Fit the classifier on train set
# - Made predictions using the classifier
# - Calculated accuracy using module provided by sklearn
# - Extended the working to varying depths of the tree
# - Visualizing using matplotlib

decision_tree = tree.DecisionTreeClassifier()
decision_tree = decision_tree.fit(X,y)

y_pred = decision_tree.predict(X_test)

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=200)
tree.plot_tree(decision_tree, filled=True)

accuracy_score(y_test, y_pred)

predicion_array = {'train_acc':[], 'test_acc':[], 'train_error':[], 'test_error':[]}
for i in range(len(X.columns)):
    decision_tree = tree.DecisionTreeClassifier(criterion = "entropy", max_depth = i+1)
    decision_tree = decision_tree.fit(X_train,y_train)
    train_pred = decision_tree.predict(X_train)
    test_pred = decision_tree.predict(X_test)
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    predicion_array['train_acc'].append(train_accuracy)
    predicion_array['test_acc'].append(test_accuracy)
    predicion_array['train_error'].append(1 - train_accuracy)
    predicion_array['test_error'].append(1 - test_accuracy)
    
plt.figure(figsize=(16,10), dpi= 80)
no_of_features = list(range(1,len(X.columns)+1))
plt.plot(no_of_features, predicion_array['train_acc'], color='tab:red', label='Train Accuracy')
plt.plot(no_of_features, predicion_array['test_acc'], color="steelblue", label='Test Accuracy')
plt.ylabel("Accuracy", fontsize=12)
plt.xlabel("Number of Features", fontsize=12)
plt.yticks(fontsize=12, alpha=.9)
plt.title("Train and Test Accuracy for varying depth of tree", fontsize=16)
plt.legend()
plt.show()

plt.figure(figsize=(16,10), dpi= 80)
no_of_features = list(range(1,len(X.columns)+1))
plt.plot(no_of_features, predicion_array['train_error'], color='tab:green', label='Train Error')
plt.plot(no_of_features, predicion_array['test_error'], color="tab:red", label='Test Error')
plt.ylabel("Error", fontsize=12)
plt.xlabel("Number of Features", fontsize=12)
plt.yticks(fontsize=12, alpha=.9)
plt.title("Train and Test Error for varying depth of tree", fontsize=16)
plt.legend()
plt.show()

# Hyperparameter Tuning
params = {'max_depth' : range(1, len(X.columns)+1)}

grid = GridSearchCV(decision_tree, param_grid = params, cv=10, verbose=1, n_jobs = 1)
grid.fit(X_train, y_train)

print(grid.best_params_)

# Visualizing with GraphViz

decision_tree = tree.DecisionTreeClassifier(criterion = "entropy")
decision_tree = decision_tree.fit(X,y)
y_pred = decision_tree.predict(X_test)
accuracy_score(y_test, y_pred)

dot = tree.export_graphviz(decision_tree, out_file=None, filled = True)
graph = graphviz.Source(dot)
graph

# Tree for Optimal Depth

decision_tree = tree.DecisionTreeClassifier(criterion = "entropy", max_depth=4)
decision_tree = decision_tree.fit(X,y)
y_pred = decision_tree.predict(X_test)
accuracy_score(y_test, y_pred)

dot = tree.export_graphviz(decision_tree, out_file=None, filled=True)
graph = graphviz.Source(dot)
graph

