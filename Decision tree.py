#RUN THIS AFTER PREPROCESSING

#Import libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, KFold, TimeSeriesSplit
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import GridSearchCV

#Starting with training the decision tree...

# Lets load the balanced temporal training and validation split
X_temp_balanced = pd.read_csv('X_train_temp.csv')
y_temp_balanced = pd.read_csv('y_train_temp.csv')
x_val_temp = pd.read_csv('x_val_temp.csv')
y_val_temp = pd.read_csv('y_val_temp.csv')

# and also load the balanced random and validation split
X_rand_balanced = pd.read_csv('X_train_rand.csv')
y_rand_balanced = pd.read_csv('y_train_rand.csv')
x_val_rand = pd.read_csv('x_val_rand.csv')
y_val_rand = pd.read_csv('y_val_rand.csv')

# Initialize the decision tree classifier to start off with
model = DecisionTreeClassifier(random_state=777)

# Set up 5-fold cross-validation for the random set 
kf = KFold(n_splits=5, shuffle=True, random_state=777)

# Perform cross-validation on the random set
random_cv_scores = cross_val_score(model, X_rand_balanced, y_rand_balanced, cv=kf, scoring='accuracy')

# Print the cross-validation results for the random set
print("Random Set - 5-Fold Cross-Validation Scores:", random_cv_scores)
print("Random Set - Mean Accuracy:", np.mean(random_cv_scores))

#For the random set the mean accuracy is around 79%
#The accuracy between 5 folds is between 78% and 80% 

# Set up time series cross-validation for the temporal dataset 
tscv = TimeSeriesSplit(n_splits=5) # 5 splits 

# Perform cross-validation on the temporal set 
temporal_cv_scores = cross_val_score(model, X_temp_balanced, y_temp_balanced, cv=tscv, scoring='accuracy')

# Print the cross-validation results for the temporal set
print("Temporal Set - Time Series Cross-Validation Scores:", temporal_cv_scores)
print("Temporal Set - Mean Accuracy:", np.mean(temporal_cv_scores))

#Mean accuracy is 77% 
#Across 5 splits the accuracy ranges from 71% to 89%, best perfomance is highet than with the random split, but lowest is lower than the random split. 
#So this predicts better in some time series than other time series. 

#Lets so a grid search
#I dont mind waiting longer time periods so choosing this over random search 
param_grid = {
    'criterion': ['gini', 'entropy'],  # primary criteria 
    'max_depth': [None, 5, 10, 15, 20],  
    'min_samples_split': [2, 5, 10], 
    'min_samples_leaf': [1, 2, 4]  
}

#Lets start with the random datasplit 
# Set up GridSearchCV
grid_search_rand = GridSearchCV(estimator=model, param_grid=param_grid, 
                                 scoring='accuracy', cv=5, verbose=1, n_jobs=-1)

# Fit the grid search 
grid_search_rand.fit(X_rand_balanced, y_rand_balanced)
print("Best Parameters (Random Set):", grid_search_rand.best_params_)
print("Best Score (Random Set):", grid_search_rand.best_score_)

#Output for random split=
#Best Parameters (Random Set): {'criterion': 'entropy', 'max_depth': 5, 'min_samples_leaf': 4, 'min_samples_split': 2}
#Best Score (Random Set): 0.714598350679797

#Now the temporal split
# Set up GridSearchCV
grid_search_temp = GridSearchCV(estimator=model, param_grid=param_grid, 
                                 scoring='accuracy', cv=5, verbose=1, n_jobs=-1)

#Fit the grid search
grid_search_temp.fit(X_temp_balanced, y_temp_balanced)

print("Best Parameters (Temporal Set):", grid_search_temp.best_params_)
print("Best Score (Temporal Set):", grid_search_temp.best_score_)

#Best Parameters (Temporal Set): {'criterion': 'entropy', 'max_depth': 5, 'min_samples_leaf': 4, 'min_samples_split': 2}
#Best Score (Temporal Set): 0.8033382245047689

#So the temporal split has better performance after hyperparameter tuning based on training set
#Lets try evaluating performance on the validation sets: 

# Random validation set evaluation
best_model_rand = grid_search_rand.best_estimator_
best_model_rand.fit(X_rand_balanced, y_rand_balanced)  
val_rand_predictions = best_model_rand.predict(x_val_rand)
val_rand_accuracy = accuracy_score(y_val_rand, val_rand_predictions)

print("Validation Accuracy (Random Set):", val_rand_accuracy)

#Validation Accuracy (Random Set): 0.7506393861892583

# Temporal validation set evaluation
best_model_temp = grid_search_temp.best_estimator_
best_model_temp.fit(X_temp_balanced, y_temp_balanced)  
val_temp_predictions = best_model_temp.predict(x_val_temp)
val_temp_accuracy = accuracy_score(y_val_temp, val_temp_predictions)

print("Validation Accuracy (Temporal Set):", val_temp_accuracy)

#Validation Accuracy (Temporal Set): 0.7229859335038363

