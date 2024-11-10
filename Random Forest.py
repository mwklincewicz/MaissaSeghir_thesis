#RUN THIS AFTER THE DECISION TREE
#Import libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold, TimeSeriesSplit
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.tree import plot_tree

# Lets load the balanced temporal training and validation split
#Changing y_columns from 2D arrays to 1D arrays using .ravel()
X_temp_balanced = pd.read_csv('X_train_temp.csv')
y_temp_balanced = pd.read_csv('y_train_temp.csv').values.ravel()  # Using .ravel() here because i got a warning 
x_val_temp = pd.read_csv('x_val_temp.csv')
y_val_temp = pd.read_csv('y_val_temp.csv').values.ravel()  # Using .ravel() here because i got a warning

# and also load the balanced random and validation split
X_rand_balanced = pd.read_csv('X_train_rand.csv')
y_rand_balanced = pd.read_csv('y_train_rand.csv').values.ravel()  # Using .ravel() here because i got a warning
x_val_rand = pd.read_csv('x_val_rand.csv')
y_val_rand = pd.read_csv('y_val_rand.csv').values.ravel()  # Using .ravel() here because i got a warning

# Starting with training the random forest...

# Initialize the random forest classifier with random_state 777 again 
model_rf = RandomForestClassifier(random_state=777, n_jobs=-1) #using all cores available


# Set up 5-fold cross-validation for the random set 
kf = KFold(n_splits=5, shuffle=True, random_state=777)

# Perform cross-validation on the random set
random_cv_scores = cross_val_score(model_rf, X_rand_balanced, y_rand_balanced, cv=kf, scoring='accuracy')

# Print the cross-validation results for the random set
print("Random Set - 5-Fold Cross-Validation Scores:", random_cv_scores)
print("Random Set - Mean Accuracy:", np.mean(random_cv_scores))

# Set up time series cross-validation for the temporal dataset
tscv = TimeSeriesSplit(n_splits=5) # splits 

# Perform cross-validation on the temporal set
temporal_cv_scores = cross_val_score(model_rf, X_temp_balanced, y_temp_balanced, cv=tscv, scoring='accuracy')

# Print the cross-validation results for the temporal set
print("Temporal Set - Time Series Cross-Validation Scores:", temporal_cv_scores)
print("Temporal Set - Mean Accuracy:", np.mean(temporal_cv_scores))

#Gridsearch gave me basd results, worse actually then the decision tree, but when i changed parameters it wasnt done running even after 14+ hours, so i might switch to random search

# GridSearch for Random Forest
param_grid_rf = {
    'n_estimators': [70, 100, 130, 150, 170, 200, 250, 300],  
    'max_depth': [None, 10, 15, 20, 30],  
    'min_samples_split': [2, 5, 10], 
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy'],
    'max_features': ['sqrt', 'log2', None]
},

# Set up GridSearchCV for Random Set
grid_search_rand_rf = GridSearchCV(estimator=model_rf, param_grid=param_grid_rf, 
                                   scoring='accuracy', cv=5, verbose=1, n_jobs=-1)

# Fit the grid search 
grid_search_rand_rf.fit(X_rand_balanced, y_rand_balanced)
print("Best Parameters (Random Set):", grid_search_rand_rf.best_params_)
print("Best Score (Random Set):", grid_search_rand_rf.best_score_)

# Set up GridSearchCV for Temporal Set
grid_search_temp_rf = GridSearchCV(estimator=model_rf, param_grid=param_grid_rf, 
                                   scoring='accuracy', cv=5, verbose=1, n_jobs=-1)

# Fit the grid search 
grid_search_temp_rf.fit(X_temp_balanced, y_temp_balanced)
print("Best Parameters (Temporal Set):", grid_search_temp_rf.best_params_)
print("Best Score (Temporal Set):", grid_search_temp_rf.best_score_)

# Evaluation on Random Validation Set
best_model_rand_rf = grid_search_rand_rf.best_estimator_
best_model_rand_rf.fit(X_rand_balanced, y_rand_balanced)
val_rand_predictions_rf = best_model_rand_rf.predict(x_val_rand)
val_rand_accuracy_rf = accuracy_score(y_val_rand, val_rand_predictions_rf)
print("Validation Accuracy (Random Set):", val_rand_accuracy_rf)

# Evaluation on Temporal Validation Set
best_model_temp_rf = grid_search_temp_rf.best_estimator_
best_model_temp_rf.fit(X_temp_balanced, y_temp_balanced)
val_temp_predictions_rf = best_model_temp_rf.predict(x_val_temp)
val_temp_accuracy_rf = accuracy_score(y_val_temp, val_temp_predictions_rf)
print("Validation Accuracy (Temporal Set):", val_temp_accuracy_rf)

# Metrics for Random Validation Set
precision_rand_rf = precision_score(y_val_rand, val_rand_predictions_rf)
recall_rand_rf = recall_score(y_val_rand, val_rand_predictions_rf)
f1_rand_rf = f1_score(y_val_rand, val_rand_predictions_rf)
roc_auc_rand_rf = roc_auc_score(y_val_rand, best_model_rand_rf.predict_proba(x_val_rand)[:, 1])
conf_matrix_rand_rf = confusion_matrix(y_val_rand, val_rand_predictions_rf)

print(f"Validation Metrics (Random Set):")
print(f"Accuracy: {val_rand_accuracy_rf}")
print(f"Precision: {precision_rand_rf}")
print(f"Recall: {recall_rand_rf}")
print(f"F1-Score: {f1_rand_rf}")
print(f"AUC-ROC: {roc_auc_rand_rf}")
print(f"Confusion Matrix:\n{conf_matrix_rand_rf}")

# Metrics for Temporal Validation Set
precision_temp_rf = precision_score(y_val_temp, val_temp_predictions_rf)
recall_temp_rf = recall_score(y_val_temp, val_temp_predictions_rf)
f1_temp_rf = f1_score(y_val_temp, val_temp_predictions_rf)
roc_auc_temp_rf = roc_auc_score(y_val_temp, best_model_temp_rf.predict_proba(x_val_temp)[:, 1])
conf_matrix_temp_rf = confusion_matrix(y_val_temp, val_temp_predictions_rf)

print(f"Validation Metrics (Temporal Set):")
print(f"Accuracy: {val_temp_accuracy_rf}")
print(f"Precision: {precision_temp_rf}")
print(f"Recall: {recall_temp_rf}")
print(f"F1-Score: {f1_temp_rf}")
print(f"AUC-ROC: {roc_auc_temp_rf}")
print(f"Confusion Matrix:\n{conf_matrix_temp_rf}")

# Plot Confusion Matrix for Random Set
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_rand_rf, annot=True, fmt="d", cmap="Blues", xticklabels=["Short", "Long"], yticklabels=["Short", "Long"])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Random Set)')
plt.show()

# Plot Confusion Matrix for Temporal Set
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_temp_rf, annot=True, fmt="d", cmap="Blues", xticklabels=["Short", "Long"], yticklabels=["Short", "Long"])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Temporal Set)')
plt.show()

# Feature importance analysis for Random Forest
feature_importances_rand_rf = best_model_rand_rf.feature_importances_
feature_importances_temp_rf = best_model_temp_rf.feature_importances_

importance_df_rand_rf = pd.DataFrame({
    'Feature': X_rand_balanced.columns,
    'Importance': feature_importances_rand_rf
}).sort_values(by='Importance', ascending=False)

importance_df_temp_rf = pd.DataFrame({
    'Feature': X_temp_balanced.columns,
    'Importance': feature_importances_temp_rf
}).sort_values(by='Importance', ascending=False)

# Filter non-zero importance for plotting
importance_df_rand_rf = importance_df_rand_rf[importance_df_rand_rf['Importance'] > 0].sort_values(by='Importance', ascending=False)
importance_df_temp_rf = importance_df_temp_rf[importance_df_temp_rf['Importance'] > 0].sort_values(by='Importance', ascending=False)

# Plot Feature Importance for Random Set
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df_rand_rf)
plt.title('Feature Importance (Random Set) - Non-Zero Importance Only')
plt.show()

# Plot Feature Importance for Temporal Set
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df_temp_rf)
plt.title('Feature Importance (Temporal Set) - Non-Zero Importance Only')
plt.show()

# Permutation Importance for Random Forest
perm_importance_rand_rf = permutation_importance(best_model_rand_rf, x_val_rand, y_val_rand, n_repeats=10, random_state=777)
perm_importance_temp_rf = permutation_importance(best_model_temp_rf, x_val_temp, y_val_temp, n_repeats=10, random_state=777)

# Get permutation importances and corresponding feature names for each split
perm_importances_rand_rf = perm_importance_rand_rf.importances_mean
perm_importances_temp_rf = perm_importance_temp_rf.importances_mean

# Filter non-zero permutation importance for plotting
perm_importance_df_rand_rf = pd.DataFrame({
    'Feature': X_rand_balanced.columns,
    'Permutation Importance': perm_importances_rand_rf
}).sort_values(by='Permutation Importance', ascending=False)

perm_importance_df_temp_rf = pd.DataFrame({
    'Feature': X_temp_balanced.columns,
    'Permutation Importance': perm_importances_temp_rf
}).sort_values(by='Permutation Importance', ascending=False)

filtered_df_rand_rf = perm_importance_df_rand_rf[perm_importance_df_rand_rf['Permutation Importance'] > 0]
filtered_df_temp_rf = perm_importance_df_temp_rf[perm_importance_df_temp_rf['Permutation Importance'] > 0]

# Plot Permutation Importance for Random Set
plt.figure(figsize=(10, 6))
sns.barplot(x='Permutation Importance', y='Feature', data=filtered_df_rand_rf)
plt.title('Permutation Importance (Random Set) - Non-Zero Importance Only')
plt.show()

# Plot Permutation Importance for Temporal Set
plt.figure(figsize=(10, 6))
sns.barplot(x='Permutation Importance', y='Feature', data=filtered_df_temp_rf)
plt.title('Permutation Importance (Temporal Set) - Non-Zero Importance Only')
plt.show()
