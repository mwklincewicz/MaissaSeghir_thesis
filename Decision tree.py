#RUN THIS AFTER PREPROCESSING

#Import libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, KFold, TimeSeriesSplit
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


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
#Best Parameters (Random Set): {'criterion': 'gini', 'max_depth': 5, 'min_samples_leaf': 4, 'min_samples_split': 2}
#Best Score (Random Set): 0.7168859297979816

#Now the temporal split
# Set up GridSearchCV
grid_search_temp = GridSearchCV(estimator=model, param_grid=param_grid, 
                                 scoring='accuracy', cv=5, verbose=1, n_jobs=-1)

#Fit the grid search
grid_search_temp.fit(X_temp_balanced, y_temp_balanced)

print("Best Parameters (Temporal Set):", grid_search_temp.best_params_)
print("Best Score (Temporal Set):", grid_search_temp.best_score_)

#Best Parameters (Temporal Set): {'criterion': 'gini', 'max_depth': 5, 'min_samples_leaf': 1, 'min_samples_split': 10}
#Best Score (Temporal Set): 0.605490187027098

#So the temporal split has worse performance after hyperparameter tuning based on training set
#Lets try evaluating performance on the validation sets: 

# Random validation set evaluation
best_model_rand = grid_search_rand.best_estimator_
best_model_rand.fit(X_rand_balanced, y_rand_balanced)  
val_rand_predictions = best_model_rand.predict(x_val_rand)
val_rand_accuracy = accuracy_score(y_val_rand, val_rand_predictions)

print("Validation Accuracy (Random Set):", val_rand_accuracy)

#Validation Accuracy (Random Set): 0.7509590792838875

# Temporal validation set evaluation
best_model_temp = grid_search_temp.best_estimator_
best_model_temp.fit(X_temp_balanced, y_temp_balanced)  
val_temp_predictions = best_model_temp.predict(x_val_temp)
val_temp_accuracy = accuracy_score(y_val_temp, val_temp_predictions)

print("Validation Accuracy (Temporal Set):", val_temp_accuracy)

#Validation Accuracy (Temporal Set): 0.9402173913043478

#METRICS
#METRICS FOR RANDOM VALIDATION SET
val_rand_predictions = best_model_rand.predict(x_val_rand)

# Calculate precision, recall, f1-score, and AUC-ROC 
precision_rand = precision_score(y_val_rand, val_rand_predictions)
recall_rand = recall_score(y_val_rand, val_rand_predictions)
f1_rand = f1_score(y_val_rand, val_rand_predictions)

# AUC-ROC 
val_rand_probabilities = best_model_rand.predict_proba(x_val_rand)[:, 1]
roc_auc_rand = roc_auc_score(y_val_rand, val_rand_probabilities)

# Confusion Matrix
conf_matrix_rand = confusion_matrix(y_val_rand, val_rand_predictions)

# Print the evaluation metrics for the random validation set... Lets see
print(f"Validation Metrics (Random Set):")
print(f"Accuracy: {val_rand_accuracy}")
print(f"Precision: {precision_rand}")
print(f"Recall: {recall_rand}")
print(f"F1-Score: {f1_rand}")
print(f"AUC-ROC: {roc_auc_rand}")
print(f"Confusion Matrix:\n{conf_matrix_rand}")

# Now lets evaluate on temporal validation set
val_temp_predictions = best_model_temp.predict(x_val_temp)

# the same metrics on this df
precision_temp = precision_score(y_val_temp, val_temp_predictions)
recall_temp = recall_score(y_val_temp, val_temp_predictions)
f1_temp = f1_score(y_val_temp, val_temp_predictions)

# AUC-ROC
val_temp_probabilities = best_model_temp.predict_proba(x_val_temp)[:, 1]
roc_auc_temp = roc_auc_score(y_val_temp, val_temp_probabilities)

# Confusion Matrix
conf_matrix_temp = confusion_matrix(y_val_temp, val_temp_predictions)

# Print evaluation metrics but now for the temporal validation set
print(f"Validation Metrics (Temporal Set):")
print(f"Accuracy: {val_temp_accuracy}")
print(f"Precision: {precision_temp}")
print(f"Recall: {recall_temp}")
print(f"F1-Score: {f1_temp}")
print(f"AUC-ROC: {roc_auc_temp}")
print(f"Confusion Matrix:\n{conf_matrix_temp}")

# Plot confusion matrix for the random set
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_rand, annot=True, fmt="d", cmap="Blues", xticklabels=["Short", "Long"], yticklabels=["Short", "Long"])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Random Set)')
plt.show()

# Plot confusion matrix for the tem[p] set
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_temp, annot=True, fmt="d", cmap="Blues", xticklabels=["Short", "Long"], yticklabels=["Short", "Long"])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Temporal Set)')
plt.show()

#Feature importance analysis
#Here we can check if the temporal stratification actually helped with getting out the temporal bias... 

# Get feature importance from the best model in both rand and temp
feature_importances_rand = best_model_rand.feature_importances_
feature_importances_temp = best_model_temp.feature_importances_

# Combine feature importances with the feature names
feature_names_rand = X_rand_balanced.columns
feature_names_temp = X_temp_balanced.columns

# Create DataFrames to visualize the feature importance
importance_df_rand = pd.DataFrame({
    'Feature': feature_names_rand,
    'Importance': feature_importances_rand
}).sort_values(by='Importance', ascending=False)

importance_df_temp = pd.DataFrame({
    'Feature': feature_names_temp,
    'Importance': feature_importances_temp
}).sort_values(by='Importance', ascending=False)

# Filter for features with non-zero importance
importance_df_rand = importance_df_rand[importance_df_rand['Importance'] > 0].sort_values(by='Importance', ascending=False)

importance_df_temp = importance_df_temp[importance_df_temp['Importance'] > 0].sort_values(by='Importance', ascending=False)

# Plot the important features
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df_rand)
plt.title('Feature Importance (Random Set) - Non-Zero Importance Only')
plt.show()


plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df_temp)
plt.title('Feature Importance (Temporal Set) - Non-Zero Importance Only')
plt.show()

#Probably should to feature engineering back to drop temporal features since the model is overfitting and it amplifies temporal bias

"""Results before dropping the temporal features such as house age and contract starting year:
Validation Metrics (Random Set):
Accuracy: 0.7509590792838875
Precision: 0.9330212459488657
Recall: 0.6537976280595509
F1-Score: 0.7688427299703265
AUC-ROC: 0.8726794039809362
Confusion Matrix:
[[2107  186]
 [1372 2591]]

 Validation Metrics (Temporal Set):
Accuracy: 0.9402173913043478
Precision: 0.9554036458333334
Recall: 0.983249581239531
F1-Score: 0.9691266303450552
AUC-ROC: 0.5324987407901981
Confusion Matrix:
[[  12  274]
 [ 100 5870]]

 And after dropping temporal features:


"""
