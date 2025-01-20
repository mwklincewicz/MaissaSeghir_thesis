# Importing libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold, TimeSeriesSplit, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, accuracy_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import optuna

# remove proxies 
features_to_remove = [
    'Totaal_Aantal_Historische_Huurders',
    'Aantal_Historische_Huurders_Vanaf_contractdatum',
    'Contract_duur_vorige_huurder_n.v.t.',
    'Contract_duur_vorige_huurder_>3',
    'Contract_duur_vorige_huurder_<=3',
    'Contract_duur_vorige_huurder_encoded',
    'Totaal_aantal_historisch_huurders_kort',
    'Totaal_aantal_historisch_huurders_lang',
    'Proportie_historisch_huurders_kort',
    'Proportie_historisch_huurders_lang',
    'Kort_Lang_Ratio',
    'Lang_Kort_Ratio',
    'Huurdersomloopsnelheid',

]

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

# Remove proxy features from the temporal datasets
X_temp_balanced = X_temp_balanced.drop(columns=features_to_remove, errors='ignore')
x_val_temp = x_val_temp.drop(columns=features_to_remove, errors='ignore')
X_test_temp = X_test_temp.drop(columns=features_to_remove, errors='ignore')

# Remove proxy features from the random datasets
X_rand_balanced = X_rand_balanced.drop(columns=features_to_remove, errors='ignore')
x_val_rand = x_val_rand.drop(columns=features_to_remove, errors='ignore')
X_test_rand = X_test_rand.drop(columns=features_to_remove, errors='ignore')


# Initialize the random forest classifier with random_state for reproducibility
model_rf = RandomForestClassifier(random_state=777, n_jobs=-1)  

# Set up 5-fold cross-validation for the random set, focusing on F1 score
kf = KFold(n_splits=5, shuffle=True, random_state=777)
random_cv_scores = cross_val_score(model_rf, X_rand_balanced, y_rand_balanced, cv=kf, scoring='f1')

# Print cross-validation results for the random set
print("Random Set - 5-Fold Cross-Validation F1 Scores:", random_cv_scores)
print("Random Set - Mean F1 Score:", np.mean(random_cv_scores))

# Custom function to create stratified time-series splits
def stratified_time_series_split(X, y, n_splits=5):
    indices = []
    stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=False)
    for train_index, val_index in stratified_kfold.split(X, y):
        indices.append((train_index, val_index))
    return indices

# Use the custom stratified time-series split function
stratified_splits = stratified_time_series_split(X_temp_balanced, y_temp_balanced, n_splits=5)

# Collect the F1 scores from each fold
temporal_cv_scores = []
for train_index, val_index in stratified_splits:
    X_train, X_val = X_temp_balanced.iloc[train_index], X_temp_balanced.iloc[val_index]
    y_train, y_val = y_temp_balanced[train_index], y_temp_balanced[val_index]
    
    # Fit the model and calculate F1 score on the validation set
    model_rf.fit(X_train, y_train)
    y_pred = model_rf.predict(X_val)
    f1_score_temp = f1_score(y_val, y_pred)
    temporal_cv_scores.append(f1_score_temp)

# Print the results
print("Temporal Set - Stratified Time Series Cross-Validation F1 Scores:", temporal_cv_scores)
print("Temporal Set - Mean F1 Score:", np.mean(temporal_cv_scores))

# Hyperparameter tuning using Optuna (Bayesian optimization)

def objective(trial):
    # Defining the hyperparameter space
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 70, 200, step=10),
        'max_depth': trial.suggest_int('max_depth', 10, 40, step=5),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    }
    
    model = RandomForestClassifier(**param, random_state=777, n_jobs=-1)
    
    # Cross-validation using F1 score
    kf = KFold(n_splits=5, shuffle=True, random_state=777)
    score = cross_val_score(model, X_rand_balanced, y_rand_balanced, cv=kf, scoring='f1').mean()
    
    return score

# Running Optuna optimization for Random Set
study_rand = optuna.create_study(direction='maximize')
study_rand.optimize(objective, n_trials=50)

print("Best Parameters (Random Set):", study_rand.best_params)
print("Best F1 Score (Random Set):", study_rand.best_value)

# Running Optuna optimization for Temporal Set
study_temp = optuna.create_study(direction='maximize')
study_temp.optimize(objective, n_trials=50)

print("Best Parameters (Temporal Set):", study_temp.best_params)
print("Best F1 Score (Temporal Set):", study_temp.best_value)

# Visualization of Optimization History
plt.figure(figsize=(10, 6))
optuna.visualization.matplotlib.plot_optimization_history(study_rand)
plt.title('Optimization History RF (Random Set)')
plt.show()

plt.figure(figsize=(10, 6))
optuna.visualization.matplotlib.plot_optimization_history(study_temp)
plt.title('Optimization History RF (Temporal Set)')
plt.show()

# Validation Metrics for Random Set
best_model_rand_rf = RandomForestClassifier(**study_rand.best_params, random_state=777, n_jobs=-1)
best_model_rand_rf.fit(X_rand_balanced, y_rand_balanced)
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
best_model_temp_rf = RandomForestClassifier(**study_temp.best_params, random_state=777, n_jobs=-1)
best_model_temp_rf.fit(X_temp_balanced, y_temp_balanced)
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

#FINAL FIT

# Combine training and validation datasets for Random Set
X_train_val_rand = pd.concat([X_rand_balanced, x_val_rand])
y_train_val_rand = np.concatenate([y_rand_balanced, y_val_rand])

# Combine training and validation datasets for Temporal Set
X_train_val_temp = pd.concat([X_temp_balanced, x_val_temp])
y_train_val_temp = np.concatenate([y_temp_balanced, y_val_temp])

# Refit the best model for the Random Set using Optuna's best params
final_model_rand_rf = RandomForestClassifier(**study_rand.best_params, random_state=777, n_jobs=-1)
final_model_rand_rf.fit(X_train_val_rand, y_train_val_rand)

# Refit the best model for the Temporal Set using Optuna's best params
final_model_temp_rf = RandomForestClassifier(**study_temp.best_params, random_state=777, n_jobs=-1)
final_model_temp_rf.fit(X_train_val_temp, y_train_val_temp)

# Permutation Importance for Random Set
perm_importance_rand_rf = permutation_importance(final_model_rand_rf, x_val_rand, y_val_rand, n_repeats=10, random_state=777)
importance_rand_df_rf = pd.DataFrame({
    'Feature': X_rand_balanced.columns,
    'Importance': perm_importance_rand_rf.importances_mean
})
importance_rand_df_rf = importance_rand_df_rf[(importance_rand_df_rf['Importance'] > 0.001) | (importance_rand_df_rf['Importance'] < -0.001)] # Cut off to avoid clutter in the plot
importance_rand_df_rf.sort_values(by='Importance', ascending=False, inplace=True)

# Plot Permutation Importance for Random Set
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_rand_df_rf)
plt.title('Permutation Importance (Random Set)')
plt.show()

# Permutation Importance for Temporal Set
perm_importance_temp_rf = permutation_importance(final_model_temp_rf, x_val_temp, y_val_temp, n_repeats=10, random_state=777)
importance_temp_df_rf = pd.DataFrame({
    'Feature': X_temp_balanced.columns,
    'Importance': perm_importance_temp_rf.importances_mean
})
importance_temp_df_rf = importance_temp_df_rf[(importance_temp_df_rf['Importance'] > 0.001) | (importance_temp_df_rf['Importance'] < -0.001)] # Cut off to avoid clutter in the plot
importance_temp_df_rf.sort_values(by='Importance', ascending=False, inplace=True)

# Plot Permutation Importance for Temporal Set
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_temp_df_rf)
plt.title('Permutation Importance (Temporal Set)')
plt.show()

# Final Test Set Evaluation for Random Set
test_rand_predictions_rf = final_model_rand_rf.predict(X_test_rand)

test_rand_f1_rf = f1_score(y_test_rand, test_rand_predictions_rf)
test_rand_accuracy_rf = accuracy_score(y_test_rand, test_rand_predictions_rf)
test_rand_precision_rf = precision_score(y_test_rand, test_rand_predictions_rf)
test_rand_recall_rf = recall_score(y_test_rand, test_rand_predictions_rf)
test_rand_roc_auc_rf = roc_auc_score(y_test_rand, final_model_rand_rf.predict_proba(X_test_rand)[:, 1])
test_rand_conf_matrix_rf = confusion_matrix(y_test_rand, test_rand_predictions_rf)

print("\nTest Metrics (Random Set):")
print(f"Accuracy: {test_rand_accuracy_rf}")
print(f"Precision: {test_rand_precision_rf}")
print(f"Recall: {test_rand_recall_rf}")
print(f"F1-Score: {test_rand_f1_rf}")
print(f"AUC-ROC: {test_rand_roc_auc_rf}")
print(f"Confusion Matrix:\n{test_rand_conf_matrix_rf}")

# Test Set Evaluation for Temporal Set
test_temp_predictions_rf = final_model_temp_rf.predict(X_test_temp)
test_temp_f1_rf = f1_score(y_test_temp, test_temp_predictions_rf)
test_temp_accuracy_rf = accuracy_score(y_test_temp, test_temp_predictions_rf)
test_temp_precision_rf = precision_score(y_test_temp, test_temp_predictions_rf)
test_temp_recall_rf = recall_score(y_test_temp, test_temp_predictions_rf)
test_temp_roc_auc_rf = roc_auc_score(y_test_temp, final_model_temp_rf.predict_proba(X_test_temp)[:, 1])
test_temp_conf_matrix_rf = confusion_matrix(y_test_temp, test_temp_predictions_rf)

print("\nTest Metrics (Temporal Set):")
print(f"Accuracy: {test_temp_accuracy_rf}")
print(f"Precision: {test_temp_precision_rf}")
print(f"Recall: {test_temp_recall_rf}")
print(f"F1-Score: {test_temp_f1_rf}")
print(f"AUC-ROC: {test_temp_roc_auc_rf}")
print(f"Confusion Matrix:\n{test_temp_conf_matrix_rf}")


"""RESULTS:

"""