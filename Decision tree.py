# Importing libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, KFold, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.tree import plot_tree

# Load the balanced temporal training, validation and test datasets
X_temp_balanced = pd.read_csv('X_train_temp.csv')
y_temp_balanced = pd.read_csv('y_train_temp.csv')
x_val_temp = pd.read_csv('x_val_temp.csv')
y_val_temp = pd.read_csv('y_val_temp.csv')
x_test_temp = pd.read_csv('x_test_temp.csv')
y_test_temp = pd.read_csv('y_test_temp.csv')

# Load the balanced random training, validation and test datasets
X_rand_balanced = pd.read_csv('X_train_rand.csv')
y_rand_balanced = pd.read_csv('y_train_rand.csv')
x_val_rand = pd.read_csv('x_val_rand.csv')
y_val_rand = pd.read_csv('y_val_rand.csv')
x_test_rand = pd.read_csv('x_test_rand.csv')
y_test_rand = pd.read_csv('y_test_rand.csv')

# Initialize the decision tree classifier with random state 777
model = DecisionTreeClassifier(random_state=777)

#Cross valication on the random set
#Im using f1 as a measurement because accuracy might show a false high score by just predicting majority class all the time
kf = KFold(n_splits=5, shuffle=True, random_state=777)
random_cv_scores = cross_val_score(model, X_rand_balanced, y_rand_balanced, cv=kf, scoring='f1')
print("Random Set - 5-Fold Cross-Validation F1 Scores:", random_cv_scores)
print("Random Set - Mean F1 Score:", np.mean(random_cv_scores))

#Cross valication on the temporal set
#Im using f1 as a measurement because accuracy might show a false high score by just predicting majority class all the time
tscv = TimeSeriesSplit(n_splits=5)
temporal_cv_scores = cross_val_score(model, X_temp_balanced, y_temp_balanced, cv=tscv, scoring='f1')
print("Temporal Set - Time Series Cross-Validation F1 Scores:", temporal_cv_scores)
print("Temporal Set - Mean F1 Score:", np.mean(temporal_cv_scores))

# Grid Search for Hyperparameter Tuning
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid Search for Random Split
grid_search_rand = GridSearchCV(estimator=model, param_grid=param_grid, 
                                 scoring='f1', cv=5, verbose=1, n_jobs=-1) #using all cores for faster processing
grid_search_rand.fit(X_rand_balanced, y_rand_balanced)
print("Best Parameters (Random Set):", grid_search_rand.best_params_)
print("Best F1 Score (Random Set):", grid_search_rand.best_score_)

# Grid Search for Temporal Split
grid_search_temp = GridSearchCV(estimator=model, param_grid=param_grid, 
                                 scoring='f1', cv=5, verbose=1, n_jobs=-1) #using all cores for faster processing
grid_search_temp.fit(X_temp_balanced, y_temp_balanced)
print("Best Parameters (Temporal Set):", grid_search_temp.best_params_)
print("Best F1 Score (Temporal Set):", grid_search_temp.best_score_)


# Random Validation Set 
best_model_rand = grid_search_rand.best_estimator_
best_model_rand.fit(X_rand_balanced, y_rand_balanced)
val_rand_predictions = best_model_rand.predict(x_val_rand)

# Calculate evaluation metrics on the validation set 
accuracy_rand = accuracy_score(y_val_rand, val_rand_predictions)
precision_rand = precision_score(y_val_rand, val_rand_predictions)
recall_rand = recall_score(y_val_rand, val_rand_predictions)
f1_rand = f1_score(y_val_rand, val_rand_predictions)
roc_auc_rand = roc_auc_score(y_val_rand, best_model_rand.predict_proba(x_val_rand)[:, 1])
conf_matrix_rand = confusion_matrix(y_val_rand, val_rand_predictions)

# Print evaluation metrics
print("Validation Metrics (Random Set):")
print(f"Accuracy: {accuracy_rand}")
print(f"Precision: {precision_rand}")
print(f"Recall: {recall_rand}")
print(f"F1-Score: {f1_rand}")
print(f"AUC-ROC: {roc_auc_rand}")
print(f"Confusion Matrix:\n{conf_matrix_rand}")

# Temporal Validation Set
best_model_temp = grid_search_temp.best_estimator_
best_model_temp.fit(X_temp_balanced, y_temp_balanced)
val_temp_predictions = best_model_temp.predict(x_val_temp)

# Calculate evaluation metrics on the temporal validation set
accuracy_temp = accuracy_score(y_val_temp, val_temp_predictions)
precision_temp = precision_score(y_val_temp, val_temp_predictions)
recall_temp = recall_score(y_val_temp, val_temp_predictions)
f1_temp = f1_score(y_val_temp, val_temp_predictions)
roc_auc_temp = roc_auc_score(y_val_temp, best_model_temp.predict_proba(x_val_temp)[:, 1])
conf_matrix_temp = confusion_matrix(y_val_temp, val_temp_predictions)

# Print evaluation metrics
print("Validation Metrics (Temporal Set):")
print(f"Accuracy: {accuracy_temp}")
print(f"Precision: {precision_temp}")
print(f"Recall: {recall_temp}")
print(f"F1-Score: {f1_temp}")
print(f"AUC-ROC: {roc_auc_temp}")
print(f"Confusion Matrix:\n{conf_matrix_temp}")


# Permutation Importance for Random Set
perm_importance_rand = permutation_importance(best_model_rand, x_val_rand, y_val_rand, n_repeats=10, random_state=777)
rand_importance_df = pd.DataFrame({
    'Feature': X_rand_balanced.columns,
    'Importance': perm_importance_rand.importances_mean
})

# I dont want a cluttered plot... So using a cutoff of 0.001 and -0.001
rand_importance_df = rand_importance_df[(rand_importance_df['Importance'] > 0.001) | (rand_importance_df['Importance'] < -0.001)]
rand_importance_df.sort_values(by='Importance', ascending=False, inplace=True)

# Plotting Permutation Importance for Random Set
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=rand_importance_df)
plt.title('Permutation Importance (Random Set)')
plt.xlabel('Mean Importance')
plt.ylabel('Feature')
plt.show()

# Permutation Importance for Temporal Set
perm_importance_temp = permutation_importance(best_model_temp, x_val_temp, y_val_temp, n_repeats=10, random_state=777)
temp_importance_df = pd.DataFrame({
    'Feature': X_temp_balanced.columns,
    'Importance': perm_importance_temp.importances_mean
})

# I dont want a cluttered plot... So using a cutoff of 0.001 and -0.001
temp_importance_df = temp_importance_df[(temp_importance_df['Importance'] > 0.001) | (temp_importance_df['Importance'] < -0.001)]
temp_importance_df.sort_values(by='Importance', ascending=False, inplace=True)

# Plotting Permutation Importance for Temporal Set 
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=temp_importance_df)
plt.title('Permutation Importance (Temporal Set)')
plt.xlabel('Mean Importance')
plt.ylabel('Feature')
plt.show()

#FINAL EVALUATION ON THE TEST SET


# Test the best model from the Random Split on the Random Test Set
test_rand_predictions = best_model_rand.predict(x_test_rand)

# Calculate the evaluation metrics from the random split on the Random Test Set
accuracy_test_rand = accuracy_score(y_test_rand, test_rand_predictions)
precision_test_rand = precision_score(y_test_rand, test_rand_predictions)
recall_test_rand = recall_score(y_test_rand, test_rand_predictions)
f1_test_rand = f1_score(y_test_rand, test_rand_predictions)
roc_auc_test_rand = roc_auc_score(y_test_rand, best_model_rand.predict_proba(x_test_rand)[:, 1])
conf_matrix_test_rand = confusion_matrix(y_test_rand, test_rand_predictions)

# Print evaluation metrics for the Random Test Set
print("\nTest Metrics (Random Set):")
print(f"Accuracy: {accuracy_test_rand:.4f}")
print(f"Precision: {precision_test_rand:.4f}")
print(f"Recall: {recall_test_rand:.4f}")
print(f"F1-Score: {f1_test_rand:.4f}")
print(f"AUC-ROC: {roc_auc_test_rand:.4f}")
print(f"Confusion Matrix:\n{conf_matrix_test_rand}")


# Test the best model from the Temporal Split on the Temporal Test Set
test_temp_predictions = best_model_temp.predict(x_test_temp)

# Calculate evaluation metrics on the Temporal Test Set
accuracy_test_temp = accuracy_score(y_test_temp, test_temp_predictions)
precision_test_temp = precision_score(y_test_temp, test_temp_predictions)
recall_test_temp = recall_score(y_test_temp, test_temp_predictions)
f1_test_temp = f1_score(y_test_temp, test_temp_predictions)
roc_auc_test_temp = roc_auc_score(y_test_temp, best_model_temp.predict_proba(x_test_temp)[:, 1])
conf_matrix_test_temp = confusion_matrix(y_test_temp, test_temp_predictions)

# Print evaluation metrics for the Temporal Test Set
print("\nTest Metrics (Temporal Set):")
print(f"Accuracy: {accuracy_test_temp:.4f}")
print(f"Precision: {precision_test_temp:.4f}")
print(f"Recall: {recall_test_temp:.4f}")
print(f"F1-Score: {f1_test_temp:.4f}")
print(f"AUC-ROC: {roc_auc_test_temp:.4f}")
print(f"Confusion Matrix:\n{conf_matrix_test_temp}")

"""RESULTS:
Random Set - 5-Fold Cross-Validation F1 Scores: [0.78583618 0.79333478 0.78772157 0.79343838 0.7962248 ]
Random Set - Mean F1 Score: 0.7913111400594282

Temporal Set - Time Series Cross-Validation F1 Scores: [0.83591423 0.83464333 0.82290722 0.36111111 0.        ]
Temporal Set - Mean F1 Score: 0.5709151804656348

Fitting 5 folds for each of 90 candidates, totalling 450 fits
Best Parameters (Random Set): {'criterion': 'entropy', 'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2}
Best F1 Score (Random Set): 0.6328442113577373

Fitting 5 folds for each of 90 candidates, totalling 450 fits
Best Parameters (Temporal Set): {'criterion': 'entropy', 'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2}
Best F1 Score (Temporal Set): 0.7882314350466181

Validation Metrics (Random Set):
Accuracy: 0.7482416879795396
Precision: 0.8124018838304553
Recall: 0.7834973504920515
F1-Score: 0.7976878612716762
AUC-ROC: 0.7422321982040812
Confusion Matrix:
[[1576  717]
 [ 858 3105]]

Validation Metrics (Temporal Set):
Accuracy: 0.6120524296675192
Precision: 0.5796265788028556
Recall: 0.7020285999334885
F1-Score: 0.6349827041660401
AUC-ROC: 0.6153170047564198
Confusion Matrix:
[[1718 1531]
 [ 896 2111]]

Test Metrics (Random Set):
Accuracy: 0.7521
Precision: 0.8159
Recall: 0.7794
F1-Score: 0.7972
AUC-ROC: 0.7478
Confusion Matrix:
[[1657  688]
 [ 863 3049]]

Test Metrics (Temporal Set):
Accuracy: 0.6014
Precision: 0.5736
Recall: 0.6986
F1-Score: 0.6300
AUC-ROC: 0.6041
Confusion Matrix:
[[1640 1578]
 [ 916 2123]]
"""