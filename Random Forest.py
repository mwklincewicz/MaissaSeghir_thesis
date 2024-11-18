# Importing libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold, TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, accuracy_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

# Load the balanced temporal training and validation datasets
X_temp_balanced = pd.read_csv('X_train_temp.csv')
y_temp_balanced = pd.read_csv('y_train_temp.csv').values.ravel()
x_val_temp = pd.read_csv('x_val_temp.csv')
y_val_temp = pd.read_csv('y_val_temp.csv').values.ravel()
X_test_temp = pd.read_csv('X_test_temp.csv')
y_test_temp = pd.read_csv('y_test_temp.csv').values.ravel()

# Load the balanced random training and validation datasets
X_rand_balanced = pd.read_csv('X_train_rand.csv')
y_rand_balanced = pd.read_csv('y_train_rand.csv').values.ravel()
x_val_rand = pd.read_csv('x_val_rand.csv')
y_val_rand = pd.read_csv('y_val_rand.csv').values.ravel()
X_test_rand = pd.read_csv('X_test_rand.csv')
y_test_rand = pd.read_csv('y_test_rand.csv').values.ravel()



# Initialize the random forest classifier with random_state for reproducibility
model_rf = RandomForestClassifier(random_state=777, n_jobs=-1)  # Using all cores available

# Set up 5-fold cross-validation for the random set, focusing on F1 score
kf = KFold(n_splits=5, shuffle=True, random_state=777)
random_cv_scores = cross_val_score(model_rf, X_rand_balanced, y_rand_balanced, cv=kf, scoring='f1')

# Print cross-validation results for the random set
print("Random Set - 5-Fold Cross-Validation F1 Scores:", random_cv_scores)
print("Random Set - Mean F1 Score:", np.mean(random_cv_scores))

# Set up time series cross-validation for the temporal dataset, focusing on F1 score
tscv = TimeSeriesSplit(n_splits=5)
temporal_cv_scores = cross_val_score(model_rf, X_temp_balanced, y_temp_balanced, cv=tscv, scoring='f1')

# Print cross-validation results for the temporal set
print("Temporal Set - Time Series Cross-Validation F1 Scores:", temporal_cv_scores)
print("Temporal Set - Mean F1 Score:", np.mean(temporal_cv_scores))

# Hyperparameter tuning using RandomizedSearchCV focused on F1 score
param_dist_rf = {
    'n_estimators': [70, 80, 130, 150, 200],
    'max_depth': [None, 20, 25, 27, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy'],
    'max_features': ['sqrt', 'log2', None]
}

# RandomizedSearchCV for Random Set
random_search_rand_rf = RandomizedSearchCV(
    estimator=model_rf,
    param_distributions=param_dist_rf,
    n_iter=50,
    scoring='f1',
    cv=5,
    verbose=1,
    n_jobs=-1,
    random_state=777
)
random_search_rand_rf.fit(X_rand_balanced, y_rand_balanced)
print("Best Parameters (Random Set):", random_search_rand_rf.best_params_)
print("Best F1 Score (Random Set):", random_search_rand_rf.best_score_)

# RandomizedSearchCV for Temporal Set
random_search_temp_rf = RandomizedSearchCV(
    estimator=model_rf,
    param_distributions=param_dist_rf,
    n_iter=50,
    scoring='f1',
    cv=5,
    verbose=1,
    n_jobs=-1,
    random_state=777
)
random_search_temp_rf.fit(X_temp_balanced, y_temp_balanced)
print("Best Parameters (Temporal Set):", random_search_temp_rf.best_params_)
print("Best F1 Score (Temporal Set):", random_search_temp_rf.best_score_)

# Validation Metrics for Random Set
best_model_rand_rf = random_search_rand_rf.best_estimator_
val_rand_predictions_rf = best_model_rand_rf.predict(x_val_rand)
val_rand_accuracy_rf = accuracy_score(y_val_rand, val_rand_predictions_rf)
val_rand_precision_rf = precision_score(y_val_rand, val_rand_predictions_rf)
val_rand_recall_rf = recall_score(y_val_rand, val_rand_predictions_rf)
val_rand_f1_rf = f1_score(y_val_rand, val_rand_predictions_rf)
val_rand_roc_auc_rf = roc_auc_score(y_val_rand, best_model_rand_rf.predict_proba(x_val_rand)[:, 1])
val_rand_conf_matrix_rf = confusion_matrix(y_val_rand, val_rand_predictions_rf)

print("\nValidation Metrics (Random Set):")
print(f"Accuracy: {val_rand_accuracy_rf}")
print(f"Precision: {val_rand_precision_rf}")
print(f"Recall: {val_rand_recall_rf}")
print(f"F1-Score: {val_rand_f1_rf}")
print(f"AUC-ROC: {val_rand_roc_auc_rf}")
print(f"Confusion Matrix:\n{val_rand_conf_matrix_rf}")

# Validation Metrics for Temporal Set
best_model_temp_rf = random_search_temp_rf.best_estimator_
val_temp_predictions_rf = best_model_temp_rf.predict(x_val_temp)
val_temp_accuracy_rf = accuracy_score(y_val_temp, val_temp_predictions_rf)
val_temp_precision_rf = precision_score(y_val_temp, val_temp_predictions_rf)
val_temp_recall_rf = recall_score(y_val_temp, val_temp_predictions_rf)
val_temp_f1_rf = f1_score(y_val_temp, val_temp_predictions_rf)
val_temp_roc_auc_rf = roc_auc_score(y_val_temp, best_model_temp_rf.predict_proba(x_val_temp)[:, 1])
val_temp_conf_matrix_rf = confusion_matrix(y_val_temp, val_temp_predictions_rf)

print("\nValidation Metrics (Temporal Set):")
print(f"Accuracy: {val_temp_accuracy_rf}")
print(f"Precision: {val_temp_precision_rf}")
print(f"Recall: {val_temp_recall_rf}")
print(f"F1-Score: {val_temp_f1_rf}")
print(f"AUC-ROC: {val_temp_roc_auc_rf}")
print(f"Confusion Matrix:\n{val_temp_conf_matrix_rf}")

#permutation importance
# Permutation Importance for Random Set
perm_importance_rand_rf = permutation_importance(best_model_rand_rf, x_val_rand, y_val_rand, n_repeats=10, random_state=777)
importance_rand_df_rf = pd.DataFrame({
    'Feature': X_rand_balanced.columns,
    'Importance': perm_importance_rand_rf.importances_mean
})
importance_rand_df_rf = importance_rand_df_rf[(importance_rand_df_rf['Importance'] > 0.001) | (importance_rand_df_rf['Importance'] < -0.001)] #cutt off to avoid clutter in the plot
importance_rand_df_rf.sort_values(by='Importance', ascending=False, inplace=True)

# Plot Permutation Importance for Random Set
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_rand_df_rf)
plt.title('Permutation Importance (Random Set)')
plt.show()

# Permutation Importance for Temporal Set
perm_importance_temp_rf = permutation_importance(best_model_temp_rf, x_val_temp, y_val_temp, n_repeats=10, random_state=777)
importance_temp_df_rf = pd.DataFrame({
    'Feature': X_temp_balanced.columns,
    'Importance': perm_importance_temp_rf.importances_mean
})
importance_temp_df_rf = importance_temp_df_rf[(importance_temp_df_rf['Importance'] > 0.001) | (importance_temp_df_rf['Importance'] < -0.001)]
importance_temp_df_rf.sort_values(by='Importance', ascending=False, inplace=True)

# Plot Permutation Importance for Temporal Set
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_temp_df_rf)
plt.title('Permutation Importance (Temporal Set)')
plt.show()


# final Test Set Evaluation for Random Set
best_model_rand_rf = random_search_rand_rf.best_estimator_
test_rand_predictions_rf = best_model_rand_rf.predict(X_test_rand)
test_rand_f1_rf = f1_score(y_test_rand, test_rand_predictions_rf)
test_rand_accuracy_rf = accuracy_score(y_test_rand, test_rand_predictions_rf)
test_rand_precision_rf = precision_score(y_test_rand, test_rand_predictions_rf)
test_rand_recall_rf = recall_score(y_test_rand, test_rand_predictions_rf)
test_rand_roc_auc_rf = roc_auc_score(y_test_rand, best_model_rand_rf.predict_proba(X_test_rand)[:, 1])
test_rand_conf_matrix_rf = confusion_matrix(y_test_rand, test_rand_predictions_rf)


print("\nTest Metrics (Random Set):")
print(f"Accuracy: {test_rand_accuracy_rf}")
print(f"Precision: {test_rand_precision_rf}")
print(f"Recall: {test_rand_recall_rf}")
print(f"F1-Score: {test_rand_f1_rf}")
print(f"AUC-ROC: {test_rand_roc_auc_rf}")
print(f"Confusion Matrix:\n{test_rand_conf_matrix_rf}")

# Test Set Evaluation for Temporal Set
best_model_temp_rf = random_search_temp_rf.best_estimator_
test_temp_predictions_rf = best_model_temp_rf.predict(X_test_temp)
test_temp_f1_rf = f1_score(y_test_temp, test_temp_predictions_rf)
test_temp_accuracy_rf = accuracy_score(y_test_temp, test_temp_predictions_rf)
test_temp_precision_rf = precision_score(y_test_temp, test_temp_predictions_rf)
test_temp_recall_rf = recall_score(y_test_temp, test_temp_predictions_rf)
test_temp_roc_auc_rf = roc_auc_score(y_test_temp, best_model_temp_rf.predict_proba(X_test_temp)[:, 1])
test_temp_conf_matrix_rf = confusion_matrix(y_test_temp, test_temp_predictions_rf)

print("\nTest Metrics (Temporal Set):")
print(f"Accuracy: {test_temp_accuracy_rf}")
print(f"Precision: {test_temp_precision_rf}")
print(f"Recall: {test_temp_recall_rf}")
print(f"F1-Score: {test_temp_f1_rf}")
print(f"AUC-ROC: {test_temp_roc_auc_rf}")
print(f"Confusion Matrix:\n{test_temp_conf_matrix_rf}")


"""
Random Set - 5-Fold Cross-Validation F1 Scores: [0.81631781 0.81883899 0.81481481 0.81442851 0.81117367]
Random Set - Mean F1 Score: 0.8151147595293343

Temporal Set - Time Series Cross-Validation F1 Scores: [0.8638576  0.86434438 0.85710187 0.37896929 0.        ]
Temporal Set - Mean F1 Score: 0.5928546260213288

Fitting 5 folds for each of 50 candidates, totalling 250 fits
Best Parameters (Random Set): {'n_estimators': 150, 'min_samples_split': 2, 'min_samples_leaf': 4, 'max_features': None, 'max_depth': None, 'criterion': 'entropy'}
Best F1 Score (Random Set): 0.6980236867024973

Fitting 5 folds for each of 50 candidates, totalling 250 fits
Best Parameters (Temporal Set): {'n_estimators': 150, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': None, 'max_depth': 40, 'criterion': 'entropy'}
Best F1 Score (Temporal Set): 0.8043758060866548

Validation F1 Score (Random Set): 0.827489122088559
Validation F1 Score (Temporal Set): 0.6828637147786083

Validation Metrics (Random Set):
Accuracy: 0.7845268542199488
Precision: 0.8395222020254479
Recall: 0.8157961140550088
F1-Score: 0.827489122088559
AUC-ROC: 0.8842503471106868
Confusion Matrix:
[[1675  618]
 [ 730 3233]]

Validation Metrics (Temporal Set):
Accuracy: 0.6473785166240409
Precision: 0.6014180805267156
Recall: 0.7898237445959428
F1-Score: 0.6828637147786083
AUC-ROC: 0.7715723433052436
Confusion Matrix:
[[1675 1574]
 [ 632 2375]]
 
Test Metrics (Random Set):
Accuracy: 0.7906344893719035
Precision: 0.8388020833333333
Recall: 0.8233640081799591
F1-Score: 0.8310113519091846
AUC-ROC: 0.8864579381793923
[[1726  619]
 [ 691 3221]]

Test Metrics (Temporal Set):
Accuracy: 0.6469554099408662
Precision: 0.6015166340508806
Recall: 0.8091477459690688
F1-Score: 0.6900519152518592
AUC-ROC: 0.7711806286250569
[[1589 1629]
 [ 580 2459]]
#Training on f1 results in more balanced scores which capture long term contracts better
"""
