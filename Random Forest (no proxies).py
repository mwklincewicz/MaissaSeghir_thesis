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
Random Set - 5-Fold Cross-Validation F1 Scores: [0.55978836 0.54797332 0.54338843 0.53089737 0.54760679]
Random Set - Mean F1 Score: 0.5459308539633263

Temporal Set - Stratified Time Series Cross-Validation F1 Scores: [0.5655737704918032, 0.5499075785582255, 0.6328093510681178, 0.6145299145299145, 0.6148055207026348]
Temporal Set - Mean F1 Score: 0.5955252270701392

[I 2025-01-20 12:23:54,070] A new study created in memory with name: no-name-427d946d-f972-480e-bb48-bc634d398906
[I 2025-01-20 12:24:01,441] Trial 0 finished with value: 0.5766466461926194 and parameters: {'n_estimators': 120, 'max_depth': 25, 'min_samples_split': 6, 'min_samples_leaf': 2, 'criterion': 'entropy', 'max_features': 'sqrt'}. Best is trial 0 with value: 0.5766466461926194.
[I 2025-01-20 12:24:08,187] Trial 1 finished with value: 0.563000452455809 and parameters: {'n_estimators': 120, 'max_depth': 40, 'min_samples_split': 5, 'min_samples_leaf': 1, 'criterion': 'entropy', 'max_features': 'log2'}. Best is trial 0 with value: 0.5766466461926194.
[I 2025-01-20 12:24:15,212] Trial 2 finished with value: 0.5789209775300211 and parameters: {'n_estimators': 140, 'max_depth': 40, 'min_samples_split': 9, 'min_samples_leaf': 3, 'criterion': 'gini', 'max_features': 'sqrt'}. Best is trial 2 with value: 0.5789209775300211.
[I 2025-01-20 12:24:22,218] Trial 3 finished with value: 0.5799559686189316 and parameters: {'n_estimators': 190, 'max_depth': 25, 'min_samples_split': 5, 'min_samples_leaf': 2, 'criterion': 'gini', 'max_features': 'log2'}. Best is trial 3 with value: 0.5799559686189316.
[I 2025-01-20 12:24:27,952] Trial 4 finished with value: 0.5888649919382367 and parameters: {'n_estimators': 160, 'max_depth': 20, 'min_samples_split': 3, 'min_samples_leaf': 2, 'criterion': 'gini', 'max_features': 'log2'}. Best is trial 4 with value: 0.5888649919382367.
[I 2025-01-20 12:24:30,986] Trial 5 finished with value: 0.6135092813819402 and parameters: {'n_estimators': 150, 'max_depth': 10, 'min_samples_split': 6, 'min_samples_leaf': 3, 'criterion': 'gini', 'max_features': 'log2'}. Best is trial 5 with value: 0.6135092813819402.
[I 2025-01-20 12:24:38,752] Trial 6 finished with value: 0.5798861894668021 and parameters: {'n_estimators': 190, 'max_depth': 40, 'min_samples_split': 9, 'min_samples_leaf': 2, 'criterion': 'entropy', 'max_features': 'log2'}. Best is trial 5 with value: 0.6135092813819402.
[I 2025-01-20 12:24:45,817] Trial 7 finished with value: 0.5844633471194914 and parameters: {'n_estimators': 130, 'max_depth': 20, 'min_samples_split': 5, 'min_samples_leaf': 3, 'criterion': 'gini', 'max_features': 'sqrt'}. Best is trial 5 with value: 0.6135092813819402.
[I 2025-01-20 12:25:00,459] Trial 8 finished with value: 0.5684171310529663 and parameters: {'n_estimators': 190, 'max_depth': 40, 'min_samples_split': 8, 'min_samples_leaf': 1, 'criterion': 'entropy', 'max_features': 'sqrt'}. Best is trial 5 with value: 0.6135092813819402.
[I 2025-01-20 12:25:07,826] Trial 9 finished with value: 0.5729284510488262 and parameters: {'n_estimators': 170, 'max_depth': 40, 'min_samples_split': 2, 'min_samples_leaf': 2, 'criterion': 'entropy', 'max_features': 'log2'}. Best is trial 5 with value: 0.6135092813819402.
[I 2025-01-20 12:25:29,353] Trial 10 finished with value: 0.5927150961208845 and parameters: {'n_estimators': 70, 'max_depth': 10, 'min_samples_split': 8, 'min_samples_leaf': 4, 'criterion': 'gini', 'max_features': None}. Best is trial 5 with value: 0.6135092813819402.
[I 2025-01-20 12:25:50,641] Trial 11 finished with value: 0.5927150961208845 and parameters: {'n_estimators': 70, 'max_depth': 10, 'min_samples_split': 7, 'min_samples_leaf': 4, 'criterion': 'gini', 'max_features': None}. Best is trial 5 with value: 0.6135092813819402.
[I 2025-01-20 12:26:12,813] Trial 12 finished with value: 0.5939688030863901 and parameters: {'n_estimators': 70, 'max_depth': 10, 'min_samples_split': 10, 'min_samples_leaf': 4, 'criterion': 'gini', 'max_features': None}. Best is trial 5 with value: 0.6135092813819402.
[I 2025-01-20 12:26:53,135] Trial 13 finished with value: 0.5796878307707202 and parameters: {'n_estimators': 100, 'max_depth': 15, 'min_samples_split': 10, 'min_samples_leaf': 4, 'criterion': 'gini', 'max_features': None}. Best is trial 5 with value: 0.6135092813819402.
[I 2025-01-20 12:27:35,892] Trial 14 finished with value: 0.5761687319251123 and parameters: {'n_estimators': 100, 'max_depth': 15, 'min_samples_split': 4, 'min_samples_leaf': 3, 'criterion': 'gini', 'max_features': None}. Best is trial 5 with value: 0.6135092813819402.
[I 2025-01-20 12:28:21,110] Trial 15 finished with value: 0.5944007971136243 and parameters: {'n_estimators': 150, 'max_depth': 10, 'min_samples_split': 7, 'min_samples_leaf': 4, 'criterion': 'gini', 'max_features': None}. Best is trial 5 with value: 0.6135092813819402.
[I 2025-01-20 12:28:26,113] Trial 16 finished with value: 0.5824068496493876 and parameters: {'n_estimators': 160, 'max_depth': 30, 'min_samples_split': 7, 'min_samples_leaf': 3, 'criterion': 'gini', 'max_features': 'log2'}. Best is trial 5 with value: 0.6135092813819402.
[I 2025-01-20 12:29:21,759] Trial 17 finished with value: 0.5777286626026077 and parameters: {'n_estimators': 140, 'max_depth': 15, 'min_samples_split': 6, 'min_samples_leaf': 4, 'criterion': 'gini', 'max_features': None}. Best is trial 5 with value: 0.6135092813819402.
[I 2025-01-20 12:29:26,406] Trial 18 finished with value: 0.5886304434514569 and parameters: {'n_estimators': 160, 'max_depth': 20, 'min_samples_split': 7, 'min_samples_leaf': 3, 'criterion': 'gini', 'max_features': 'log2'}. Best is trial 5 with value: 0.6135092813819402.
[I 2025-01-20 12:30:33,872] Trial 19 finished with value: 0.5657260736963632 and parameters: {'n_estimators': 150, 'max_depth': 30, 'min_samples_split': 4, 'min_samples_leaf': 4, 'criterion': 'gini', 'max_features': None}. Best is trial 5 with value: 0.6135092813819402.
[I 2025-01-20 12:31:40,261] Trial 20 finished with value: 0.5954388881203877 and parameters: {'n_estimators': 180, 'max_depth': 10, 'min_samples_split': 8, 'min_samples_leaf': 3, 'criterion': 'gini', 'max_features': None}. Best is trial 5 with value: 0.6135092813819402.
[I 2025-01-20 12:32:42,164] Trial 21 finished with value: 0.5954388881203877 and parameters: {'n_estimators': 180, 'max_depth': 10, 'min_samples_split': 8, 'min_samples_leaf': 3, 'criterion': 'gini', 'max_features': None}. Best is trial 5 with value: 0.6135092813819402.
[I 2025-01-20 12:34:09,547] Trial 22 finished with value: 0.5782115633147435 and parameters: {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 8, 'min_samples_leaf': 3, 'criterion': 'gini', 'max_features': None}. Best is trial 5 with value: 0.6135092813819402.
[I 2025-01-20 12:35:04,458] Trial 23 finished with value: 0.5936503475142919 and parameters: {'n_estimators': 180, 'max_depth': 10, 'min_samples_split': 9, 'min_samples_leaf': 3, 'criterion': 'gini', 'max_features': None}. Best is trial 5 with value: 0.6135092813819402.
[I 2025-01-20 12:36:10,841] Trial 24 finished with value: 0.5783212492296574 and parameters: {'n_estimators': 170, 'max_depth': 15, 'min_samples_split': 8, 'min_samples_leaf': 3, 'criterion': 'gini', 'max_features': None}. Best is trial 5 with value: 0.6135092813819402.
[I 2025-01-20 12:36:14,373] Trial 25 finished with value: 0.6122855509810206 and parameters: {'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 6, 'min_samples_leaf': 3, 'criterion': 'gini', 'max_features': 'log2'}. Best is trial 5 with value: 0.6135092813819402.
[I 2025-01-20 12:36:20,438] Trial 26 finished with value: 0.5922840090768046 and parameters: {'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 6, 'min_samples_leaf': 2, 'criterion': 'entropy', 'max_features': 'log2'}. Best is trial 5 with value: 0.6135092813819402.
[I 2025-01-20 12:36:24,595] Trial 27 finished with value: 0.6066050608752898 and parameters: {'n_estimators': 180, 'max_depth': 15, 'min_samples_split': 4, 'min_samples_leaf': 3, 'criterion': 'gini', 'max_features': 'log2'}. Best is trial 5 with value: 0.6135092813819402.
[I 2025-01-20 12:36:29,237] Trial 28 finished with value: 0.6060954097499823 and parameters: {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 4, 'min_samples_leaf': 3, 'criterion': 'gini', 'max_features': 'log2'}. Best is trial 5 with value: 0.6135092813819402.
[I 2025-01-20 12:36:35,707] Trial 29 finished with value: 0.5779160253275973 and parameters: {'n_estimators': 170, 'max_depth': 30, 'min_samples_split': 3, 'min_samples_leaf': 2, 'criterion': 'entropy', 'max_features': 'log2'}. Best is trial 5 with value: 0.6135092813819402.
[I 2025-01-20 12:36:40,803] Trial 30 finished with value: 0.5917168145051213 and parameters: {'n_estimators': 150, 'max_depth': 20, 'min_samples_split': 6, 'min_samples_leaf': 1, 'criterion': 'gini', 'max_features': 'log2'}. Best is trial 5 with value: 0.6135092813819402.
[I 2025-01-20 12:36:45,366] Trial 31 finished with value: 0.6060954097499823 and parameters: {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 4, 'min_samples_leaf': 3, 'criterion': 'gini', 'max_features': 'log2'}. Best is trial 5 with value: 0.6135092813819402.
[I 2025-01-20 12:36:49,883] Trial 32 finished with value: 0.6064236461896038 and parameters: {'n_estimators': 190, 'max_depth': 15, 'min_samples_split': 5, 'min_samples_leaf': 3, 'criterion': 'gini', 'max_features': 'log2'}. Best is trial 5 with value: 0.6135092813819402.
[I 2025-01-20 12:36:53,211] Trial 33 finished with value: 0.6119683482684073 and parameters: {'n_estimators': 190, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 3, 'criterion': 'gini', 'max_features': 'log2'}. Best is trial 5 with value: 0.6135092813819402.
[I 2025-01-20 12:36:56,429] Trial 34 finished with value: 0.6129837329719882 and parameters: {'n_estimators': 180, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 3, 'criterion': 'gini', 'max_features': 'log2'}. Best is trial 5 with value: 0.6135092813819402.
[I 2025-01-20 12:36:58,648] Trial 35 finished with value: 0.6137549602137382 and parameters: {'n_estimators': 120, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 3, 'criterion': 'gini', 'max_features': 'log2'}. Best is trial 35 with value: 0.6137549602137382.
[I 2025-01-20 12:37:00,832] Trial 36 finished with value: 0.6174294539513967 and parameters: {'n_estimators': 110, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2, 'criterion': 'entropy', 'max_features': 'log2'}. Best is trial 36 with value: 0.6174294539513967.
[I 2025-01-20 12:37:07,405] Trial 37 finished with value: 0.5654839443070576 and parameters: {'n_estimators': 110, 'max_depth': 35, 'min_samples_split': 5, 'min_samples_leaf': 2, 'criterion': 'entropy', 'max_features': 'sqrt'}. Best is trial 36 with value: 0.6174294539513967.
[I 2025-01-20 12:37:13,251] Trial 38 finished with value: 0.5717799151595093 and parameters: {'n_estimators': 120, 'max_depth': 25, 'min_samples_split': 3, 'min_samples_leaf': 1, 'criterion': 'entropy', 'max_features': 'log2'}. Best is trial 36 with value: 0.6174294539513967.
[I 2025-01-20 12:37:15,705] Trial 39 finished with value: 0.6169470062798724 and parameters: {'n_estimators': 130, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2, 'criterion': 'entropy', 'max_features': 'log2'}. Best is trial 36 with value: 0.6174294539513967.
[I 2025-01-20 12:37:22,846] Trial 40 finished with value: 0.5770260332765403 and parameters: {'n_estimators': 130, 'max_depth': 25, 'min_samples_split': 6, 'min_samples_leaf': 2, 'criterion': 'entropy', 'max_features': 'sqrt'}. Best is trial 36 with value: 0.6174294539513967.
[I 2025-01-20 12:37:25,018] Trial 41 finished with value: 0.6174294539513967 and parameters: {'n_estimators': 110, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2, 'criterion': 'entropy', 'max_features': 'log2'}. Best is trial 36 with value: 0.6174294539513967.
[I 2025-01-20 12:37:27,174] Trial 42 finished with value: 0.6174294539513967 and parameters: {'n_estimators': 110, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2, 'criterion': 'entropy', 'max_features': 'log2'}. Best is trial 36 with value: 0.6174294539513967.
[I 2025-01-20 12:37:29,002] Trial 43 finished with value: 0.6170309088429995 and parameters: {'n_estimators': 90, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2, 'criterion': 'entropy', 'max_features': 'log2'}. Best is trial 36 with value: 0.6174294539513967.
[I 2025-01-20 12:37:30,830] Trial 44 finished with value: 0.6164564995595877 and parameters: {'n_estimators': 90, 'max_depth': 10, 'min_samples_split': 3, 'min_samples_leaf': 2, 'criterion': 'entropy', 'max_features': 'log2'}. Best is trial 36 with value: 0.6174294539513967.
[I 2025-01-20 12:37:32,627] Trial 45 finished with value: 0.6170309088429995 and parameters: {'n_estimators': 90, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2, 'criterion': 'entropy', 'max_features': 'log2'}. Best is trial 36 with value: 0.6174294539513967.
[I 2025-01-20 12:37:35,064] Trial 46 finished with value: 0.6074375834117454 and parameters: {'n_estimators': 80, 'max_depth': 15, 'min_samples_split': 4, 'min_samples_leaf': 1, 'criterion': 'entropy', 'max_features': 'log2'}. Best is trial 36 with value: 0.6174294539513967.
[I 2025-01-20 12:37:37,049] Trial 47 finished with value: 0.6154634644339183 and parameters: {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2, 'criterion': 'entropy', 'max_features': 'log2'}. Best is trial 36 with value: 0.6174294539513967.
[I 2025-01-20 12:37:40,879] Trial 48 finished with value: 0.599784824890856 and parameters: {'n_estimators': 90, 'max_depth': 15, 'min_samples_split': 4, 'min_samples_leaf': 2, 'criterion': 'entropy', 'max_features': 'sqrt'}. Best is trial 36 with value: 0.6174294539513967.
[I 2025-01-20 12:37:43,067] Trial 49 finished with value: 0.6174294539513967 and parameters: {'n_estimators': 110, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2, 'criterion': 'entropy', 'max_features': 'log2'}. Best is trial 36 with value: 0.6174294539513967.
Best Parameters (Random Set): {'n_estimators': 110, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2, 'criterion': 'entropy', 'max_features': 'log2'}
Best F1 Score (Random Set): 0.6174294539513967

[I 2025-01-20 12:37:43,067] A new study created in memory with name: no-name-89d83dcb-58f7-4d44-b5c1-0b19b49245a1
[I 2025-01-20 12:37:49,319] Trial 0 finished with value: 0.5695696591261374 and parameters: {'n_estimators': 100, 'max_depth': 20, 'min_samples_split': 2, 'min_samples_leaf': 1, 'criterion': 'entropy', 'max_features': 'sqrt'}. Best is trial 0 with value: 0.5695696591261374.
[I 2025-01-20 12:38:38,809] Trial 1 finished with value: 0.5662286451435806 and parameters: {'n_estimators': 90, 'max_depth': 20, 'min_samples_split': 6, 'min_samples_leaf': 1, 'criterion': 'entropy', 'max_features': None}. Best is trial 0 with value: 0.5695696591261374.
[I 2025-01-20 12:38:43,591] Trial 2 finished with value: 0.5903671668670093 and parameters: {'n_estimators': 160, 'max_depth': 25, 'min_samples_split': 8, 'min_samples_leaf': 3, 'criterion': 'entropy', 'max_features': 'log2'}. Best is trial 2 with value: 0.5903671668670093.
[I 2025-01-20 12:38:50,296] Trial 3 finished with value: 0.5769036921921583 and parameters: {'n_estimators': 140, 'max_depth': 30, 'min_samples_split': 4, 'min_samples_leaf': 4, 'criterion': 'entropy', 'max_features': 'sqrt'}. Best is trial 2 with value: 0.5903671668670093.
[I 2025-01-20 12:39:00,624] Trial 4 finished with value: 0.558559414504019 and parameters: {'n_estimators': 190, 'max_depth': 35, 'min_samples_split': 3, 'min_samples_leaf': 1, 'criterion': 'gini', 'max_features': 'log2'}. Best is trial 2 with value: 0.5903671668670093.
[I 2025-01-20 12:39:06,211] Trial 5 finished with value: 0.5975003972286002 and parameters: {'n_estimators': 140, 'max_depth': 15, 'min_samples_split': 5, 'min_samples_leaf': 2, 'criterion': 'gini', 'max_features': 'sqrt'}. Best is trial 5 with value: 0.5975003972286002.
[I 2025-01-20 12:39:10,055] Trial 6 finished with value: 0.5779236363573229 and parameters: {'n_estimators': 80, 'max_depth': 25, 'min_samples_split': 4, 'min_samples_leaf': 1, 'criterion': 'entropy', 'max_features': 'log2'}. Best is trial 5 with value: 0.5975003972286002.
[I 2025-01-20 12:39:14,363] Trial 7 finished with value: 0.6107841230963207 and parameters: {'n_estimators': 150, 'max_depth': 10, 'min_samples_split': 8, 'min_samples_leaf': 1, 'criterion': 'entropy', 'max_features': 'sqrt'}. Best is trial 7 with value: 0.6107841230963207.
[I 2025-01-20 12:39:17,147] Trial 8 finished with value: 0.6030338278693119 and parameters: {'n_estimators': 100, 'max_depth': 15, 'min_samples_split': 3, 'min_samples_leaf': 2, 'criterion': 'gini', 'max_features': 'log2'}. Best is trial 7 with value: 0.6107841230963207.
[I 2025-01-20 12:39:20,897] Trial 9 finished with value: 0.5800344455303338 and parameters: {'n_estimators': 80, 'max_depth': 20, 'min_samples_split': 8, 'min_samples_leaf': 2, 'criterion': 'gini', 'max_features': 'sqrt'}. Best is trial 7 with value: 0.6107841230963207.
[I 2025-01-20 12:40:26,181] Trial 10 finished with value: 0.6006943156443828 and parameters: {'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 10, 'min_samples_leaf': 3, 'criterion': 'entropy', 'max_features': None}. Best is trial 7 with value: 0.6107841230963207.
[I 2025-01-20 12:40:28,326] Trial 11 finished with value: 0.6158994988311535 and parameters: {'n_estimators': 110, 'max_depth': 10, 'min_samples_split': 8, 'min_samples_leaf': 2, 'criterion': 'gini', 'max_features': 'log2'}. Best is trial 11 with value: 0.6158994988311535.
[I 2025-01-20 12:40:30,775] Trial 12 finished with value: 0.6173341617180468 and parameters: {'n_estimators': 120, 'max_depth': 10, 'min_samples_split': 8, 'min_samples_leaf': 2, 'criterion': 'gini', 'max_features': 'log2'}. Best is trial 12 with value: 0.6173341617180468.
[I 2025-01-20 12:40:34,538] Trial 13 finished with value: 0.5868834421015926 and parameters: {'n_estimators': 120, 'max_depth': 40, 'min_samples_split': 10, 'min_samples_leaf': 3, 'criterion': 'gini', 'max_features': 'log2'}. Best is trial 12 with value: 0.6173341617180468.
[I 2025-01-20 12:40:37,024] Trial 14 finished with value: 0.6167627549906557 and parameters: {'n_estimators': 120, 'max_depth': 10, 'min_samples_split': 7, 'min_samples_leaf': 2, 'criterion': 'gini', 'max_features': 'log2'}. Best is trial 12 with value: 0.6173341617180468.
[I 2025-01-20 12:40:41,546] Trial 15 finished with value: 0.6034388252131293 and parameters: {'n_estimators': 170, 'max_depth': 15, 'min_samples_split': 7, 'min_samples_leaf': 2, 'criterion': 'gini', 'max_features': 'log2'}. Best is trial 12 with value: 0.6173341617180468.
[I 2025-01-20 12:40:43,880] Trial 16 finished with value: 0.6139395935549692 and parameters: {'n_estimators': 120, 'max_depth': 10, 'min_samples_split': 6, 'min_samples_leaf': 4, 'criterion': 'gini', 'max_features': 'log2'}. Best is trial 12 with value: 0.6173341617180468.
[I 2025-01-20 12:41:36,769] Trial 17 finished with value: 0.5798066109059613 and parameters: {'n_estimators': 120, 'max_depth': 15, 'min_samples_split': 9, 'min_samples_leaf': 3, 'criterion': 'gini', 'max_features': None}. Best is trial 12 with value: 0.6173341617180468.
[I 2025-01-20 12:41:39,397] Trial 18 finished with value: 0.5767116659690474 and parameters: {'n_estimators': 70, 'max_depth': 30, 'min_samples_split': 7, 'min_samples_leaf': 2, 'criterion': 'gini', 'max_features': 'log2'}. Best is trial 12 with value: 0.6173341617180468.
[I 2025-01-20 12:41:43,407] Trial 19 finished with value: 0.5895697546834486 and parameters: {'n_estimators': 130, 'max_depth': 20, 'min_samples_split': 7, 'min_samples_leaf': 3, 'criterion': 'gini', 'max_features': 'log2'}. Best is trial 12 with value: 0.6173341617180468.
[I 2025-01-20 12:42:34,238] Trial 20 finished with value: 0.5952779795674812 and parameters: {'n_estimators': 170, 'max_depth': 10, 'min_samples_split': 9, 'min_samples_leaf': 2, 'criterion': 'gini', 'max_features': None}. Best is trial 12 with value: 0.6173341617180468.
[I 2025-01-20 12:42:36,625] Trial 21 finished with value: 0.6155421619060365 and parameters: {'n_estimators': 110, 'max_depth': 10, 'min_samples_split': 9, 'min_samples_leaf': 2, 'criterion': 'gini', 'max_features': 'log2'}. Best is trial 12 with value: 0.6173341617180468.
[I 2025-01-20 12:42:39,590] Trial 22 finished with value: 0.6009613612548508 and parameters: {'n_estimators': 110, 'max_depth': 15, 'min_samples_split': 6, 'min_samples_leaf': 2, 'criterion': 'gini', 'max_features': 'log2'}. Best is trial 12 with value: 0.6173341617180468.
[I 2025-01-20 12:42:42,439] Trial 23 finished with value: 0.6173032705336665 and parameters: {'n_estimators': 130, 'max_depth': 10, 'min_samples_split': 8, 'min_samples_leaf': 2, 'criterion': 'gini', 'max_features': 'log2'}. Best is trial 12 with value: 0.6173341617180468.
[I 2025-01-20 12:42:45,440] Trial 24 finished with value: 0.603144239040132 and parameters: {'n_estimators': 130, 'max_depth': 15, 'min_samples_split': 7, 'min_samples_leaf': 3, 'criterion': 'gini', 'max_features': 'log2'}. Best is trial 12 with value: 0.6173341617180468.
[I 2025-01-20 12:42:48,128] Trial 25 finished with value: 0.615428888690022 and parameters: {'n_estimators': 150, 'max_depth': 10, 'min_samples_split': 9, 'min_samples_leaf': 2, 'criterion': 'gini', 'max_features': 'log2'}. Best is trial 12 with value: 0.6173341617180468.
[I 2025-01-20 12:42:51,693] Trial 26 finished with value: 0.6053204682418093 and parameters: {'n_estimators': 130, 'max_depth': 15, 'min_samples_split': 6, 'min_samples_leaf': 1, 'criterion': 'gini', 'max_features': 'log2'}. Best is trial 12 with value: 0.6173341617180468.
[I 2025-01-20 12:42:54,237] Trial 27 finished with value: 0.6170429052995148 and parameters: {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2, 'criterion': 'gini', 'max_features': 'log2'}. Best is trial 12 with value: 0.6173341617180468.
[I 2025-01-20 12:43:41,030] Trial 28 finished with value: 0.5647827316075295 and parameters: {'n_estimators': 90, 'max_depth': 20, 'min_samples_split': 5, 'min_samples_leaf': 3, 'criterion': 'gini', 'max_features': None}. Best is trial 12 with value: 0.6173341617180468.
[I 2025-01-20 12:43:45,249] Trial 29 finished with value: 0.5878834935159049 and parameters: {'n_estimators': 100, 'max_depth': 20, 'min_samples_split': 5, 'min_samples_leaf': 1, 'criterion': 'gini', 'max_features': 'log2'}. Best is trial 12 with value: 0.6173341617180468.
[I 2025-01-20 12:43:53,072] Trial 30 finished with value: 0.5488218982528663 and parameters: {'n_estimators': 90, 'max_depth': 30, 'min_samples_split': 2, 'min_samples_leaf': 1, 'criterion': 'gini', 'max_features': 'sqrt'}. Best is trial 12 with value: 0.6173341617180468.
[I 2025-01-20 12:43:55,480] Trial 31 finished with value: 0.6167627549906557 and parameters: {'n_estimators': 120, 'max_depth': 10, 'min_samples_split': 7, 'min_samples_leaf': 2, 'criterion': 'gini', 'max_features': 'log2'}. Best is trial 12 with value: 0.6173341617180468.
[I 2025-01-20 12:43:57,481] Trial 32 finished with value: 0.6151911702592656 and parameters: {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 8, 'min_samples_leaf': 2, 'criterion': 'gini', 'max_features': 'log2'}. Best is trial 12 with value: 0.6173341617180468.
[I 2025-01-20 12:44:00,630] Trial 33 finished with value: 0.6009613612548508 and parameters: {'n_estimators': 110, 'max_depth': 15, 'min_samples_split': 6, 'min_samples_leaf': 2, 'criterion': 'gini', 'max_features': 'log2'}. Best is trial 12 with value: 0.6173341617180468.
[I 2025-01-20 12:44:03,522] Trial 34 finished with value: 0.6163251056597432 and parameters: {'n_estimators': 140, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2, 'criterion': 'entropy', 'max_features': 'log2'}. Best is trial 12 with value: 0.6173341617180468.
[I 2025-01-20 12:44:07,629] Trial 35 finished with value: 0.607121036277978 and parameters: {'n_estimators': 150, 'max_depth': 15, 'min_samples_split': 8, 'min_samples_leaf': 1, 'criterion': 'gini', 'max_features': 'log2'}. Best is trial 12 with value: 0.6173341617180468.
[I 2025-01-20 12:44:10,440] Trial 36 finished with value: 0.6153664984070748 and parameters: {'n_estimators': 130, 'max_depth': 10, 'min_samples_split': 4, 'min_samples_leaf': 2, 'criterion': 'entropy', 'max_features': 'log2'}. Best is trial 12 with value: 0.6173341617180468.
[I 2025-01-20 12:45:01,111] Trial 37 finished with value: 0.5664276117536298 and parameters: {'n_estimators': 100, 'max_depth': 25, 'min_samples_split': 7, 'min_samples_leaf': 3, 'criterion': 'gini', 'max_features': None}. Best is trial 12 with value: 0.6173341617180468.
[I 2025-01-20 12:45:08,688] Trial 38 finished with value: 0.574932272025698 and parameters: {'n_estimators': 140, 'max_depth': 40, 'min_samples_split': 6, 'min_samples_leaf': 4, 'criterion': 'entropy', 'max_features': 'sqrt'}. Best is trial 12 with value: 0.6173341617180468.
[I 2025-01-20 12:45:14,312] Trial 39 finished with value: 0.5800877654777091 and parameters: {'n_estimators': 160, 'max_depth': 35, 'min_samples_split': 8, 'min_samples_leaf': 2, 'criterion': 'gini', 'max_features': 'log2'}. Best is trial 12 with value: 0.6173341617180468.
[I 2025-01-20 12:45:18,103] Trial 40 finished with value: 0.6031562152355193 and parameters: {'n_estimators': 120, 'max_depth': 15, 'min_samples_split': 4, 'min_samples_leaf': 1, 'criterion': 'gini', 'max_features': 'log2'}. Best is trial 12 with value: 0.6173341617180468.
[I 2025-01-20 12:45:20,521] Trial 41 finished with value: 0.6167627549906557 and parameters: {'n_estimators': 120, 'max_depth': 10, 'min_samples_split': 7, 'min_samples_leaf': 2, 'criterion': 'gini', 'max_features': 'log2'}. Best is trial 12 with value: 0.6173341617180468.
[I 2025-01-20 12:45:23,006] Trial 42 finished with value: 0.6167627549906557 and parameters: {'n_estimators': 120, 'max_depth': 10, 'min_samples_split': 7, 'min_samples_leaf': 2, 'criterion': 'gini', 'max_features': 'log2'}. Best is trial 12 with value: 0.6173341617180468.
[I 2025-01-20 12:45:25,390] Trial 43 finished with value: 0.6158994988311535 and parameters: {'n_estimators': 110, 'max_depth': 10, 'min_samples_split': 8, 'min_samples_leaf': 2, 'criterion': 'gini', 'max_features': 'log2'}. Best is trial 12 with value: 0.6173341617180468.
[I 2025-01-20 12:45:27,892] Trial 44 finished with value: 0.6152603291748516 and parameters: {'n_estimators': 80, 'max_depth': 10, 'min_samples_split': 9, 'min_samples_leaf': 2, 'criterion': 'gini', 'max_features': 'sqrt'}. Best is trial 12 with value: 0.6173341617180468.
[I 2025-01-20 12:45:31,520] Trial 45 finished with value: 0.6062356888318413 and parameters: {'n_estimators': 130, 'max_depth': 15, 'min_samples_split': 6, 'min_samples_leaf': 2, 'criterion': 'entropy', 'max_features': 'log2'}. Best is trial 12 with value: 0.6173341617180468.
[I 2025-01-20 12:45:34,353] Trial 46 finished with value: 0.6168757877915608 and parameters: {'n_estimators': 140, 'max_depth': 10, 'min_samples_split': 7, 'min_samples_leaf': 2, 'criterion': 'gini', 'max_features': 'log2'}. Best is trial 12 with value: 0.6173341617180468.
[I 2025-01-20 12:45:39,318] Trial 47 finished with value: 0.6040449964356912 and parameters: {'n_estimators': 150, 'max_depth': 15, 'min_samples_split': 3, 'min_samples_leaf': 1, 'criterion': 'gini', 'max_features': 'log2'}. Best is trial 12 with value: 0.6173341617180468.
[I 2025-01-20 12:45:42,048] Trial 48 finished with value: 0.6158673367421349 and parameters: {'n_estimators': 140, 'max_depth': 10, 'min_samples_split': 8, 'min_samples_leaf': 3, 'criterion': 'gini', 'max_features': 'log2'}. Best is trial 12 with value: 0.6173341617180468.
[I 2025-01-20 12:45:51,366] Trial 49 finished with value: 0.5688467608209659 and parameters: {'n_estimators': 160, 'max_depth': 25, 'min_samples_split': 5, 'min_samples_leaf': 2, 'criterion': 'gini', 'max_features': 'sqrt'}. Best is trial 12 with value: 0.6173341617180468.
Best Parameters (Temporal Set): {'n_estimators': 120, 'max_depth': 10, 'min_samples_split': 8, 'min_samples_leaf': 2, 'criterion': 'gini', 'max_features': 'log2'}
Best F1 Score (Temporal Set): 0.6173341617180468


Validation Metrics (Random Set):
Accuracy: 0.5966010733452594
Precision: 0.5663520266518601
Recall: 0.6407035175879398
F1-Score: 0.6012378426171531
AUC-ROC: 0.6481394272725718
Confusion Matrix:
[[ 981  781]
 [ 572 1020]]

Validation Metrics (Temporal Set):
Accuracy: 0.5438282647584973
Precision: 0.4473806212331943
Recall: 0.740598618572525
F1-Score: 0.5578034682080926
AUC-ROC: 0.6228277915458195
Confusion Matrix:
[[ 859 1192]
 [ 338  965]]

Test Metrics (Random Set):
Accuracy: 0.6016696481812761
Precision: 0.5498559077809798
Recall: 0.6322067594433399
F1-Score: 0.5881627620221949
AUC-ROC: 0.6518428723054626
Confusion Matrix:
[[1064  781]
 [ 555  954]]

Test Metrics (Temporal Set):
Accuracy: 0.6607036374478235
Precision: 0.6228373702422145
Recall: 0.3997039230199852
F1-Score: 0.48692515779981965
AUC-ROC: 0.7117096376161147
Confusion Matrix:
[[1676  327]
 [ 811  540]]
"""