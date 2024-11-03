#Import libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, KFold, TimeSeriesSplit
from sklearn.metrics import accuracy_score
import numpy as np

#Starting with training the decision tree...

# Lets load the balanced temporal split
X_temp_balanced = pd.read_csv('X_temp_balanced.csv')
y_temp_balanced = pd.read_csv('y_temp_balanced.csv')

# and also load the balanced random split
X_rand_balanced = pd.read_csv('X_rand_balanced.csv')
y_rand_balanced = pd.read_csv('y_rand_balanced.csv')

# Initialize the decision tree classifier to start off with
model = DecisionTreeClassifier(random_state=0)

# Set up 5-fold cross-validation for the random set 
kf = KFold(n_splits=5, shuffle=True, random_state=0)

# Perform cross-validation on the random set
random_cv_scores = cross_val_score(model, X_rand_balanced, y_rand_balanced, cv=kf, scoring='accuracy')

# Print the cross-validation results for the random set
print("Random Set - 5-Fold Cross-Validation Scores:", random_cv_scores)
print("Random Set - Mean Accuracy:", np.mean(random_cv_scores))