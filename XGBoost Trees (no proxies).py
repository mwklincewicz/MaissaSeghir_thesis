# Importing necessary libraries
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, KFold, TimeSeriesSplit, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
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
x_temp_balanced = pd.read_csv('X_train_temp.csv')
y_temp_balanced = pd.read_csv('y_train_temp.csv')
x_val_temp = pd.read_csv('x_val_temp.csv')
y_val_temp = pd.read_csv('y_val_temp.csv')
x_test_temp = pd.read_csv('X_test_temp.csv')
y_test_temp = pd.read_csv('y_test_temp.csv')

# Load the balanced random training and validation datasets
x_rand_balanced = pd.read_csv('X_train_rand.csv')
y_rand_balanced = pd.read_csv('y_train_rand.csv')
x_val_rand = pd.read_csv('x_val_rand.csv')
y_val_rand = pd.read_csv('y_val_rand.csv')
x_test_rand = pd.read_csv('X_test_rand.csv')
y_test_rand = pd.read_csv('y_test_rand.csv')

# Remove proxy features from the temporal datasets
x_temp_balanced = x_temp_balanced.drop(columns=features_to_remove, errors='ignore')
x_val_temp = x_val_temp.drop(columns=features_to_remove, errors='ignore')
x_test_temp = x_test_temp.drop(columns=features_to_remove, errors='ignore')

# Remove proxy features from the random datasets
x_rand_balanced = x_rand_balanced.drop(columns=features_to_remove, errors='ignore')
x_val_rand = x_val_rand.drop(columns=features_to_remove, errors='ignore')
x_test_rand = x_test_rand.drop(columns=features_to_remove, errors='ignore')

# Initialize the XGBoost classifier with randomstate 777
xgb_model = XGBClassifier(random_state=777, eval_metric='logloss') 

# Set up 5-fold cross-validation for the random set
kf = KFold(n_splits=5, shuffle=True, random_state=777)

# Perform cross-validation on the random set
random_cv_scores = cross_val_score(xgb_model, x_rand_balanced, y_rand_balanced, cv=kf, scoring='f1')
print("Random Set - 5-Fold Cross-Validation F1 Scores:", random_cv_scores)
print("Random Set - Mean F1 Score:", np.mean(random_cv_scores))

# Custom function to create stratified time-series splits
def stratified_time_series_split(X, y, n_splits=5):
    # Create a list of indices to hold the splits
    indices = []
    
    # Initialize the StratifiedKFold
    stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=False)

    # Generate the indices for stratified time-series split
    for train_index, val_index in stratified_kfold.split(X, y):
        # Ensure the maintainance in the temporal order
        #slice the indices based on time (first train, then test)
        indices.append((train_index, val_index))
        
    return indices

# Use the custom stratified time-series split function
stratified_splits = stratified_time_series_split(x_temp_balanced, y_temp_balanced, n_splits=5)

# Collect the F1 scores from each fold
temporal_cv_scores = []
for train_index, val_index in stratified_splits:
    x_train, X_val = x_temp_balanced.iloc[train_index], x_temp_balanced.iloc[val_index]
    y_train, y_val = y_temp_balanced.iloc[train_index], y_temp_balanced.iloc[val_index]
    
    # Fit the model and calculate F1 score on the validation set
    xgb_model.fit(x_train, y_train)
    y_pred = xgb_model.predict(X_val)
    
    # Calculate the F1 score for this fold
    f1_score_temp = f1_score(y_val, y_pred)
    temporal_cv_scores.append(f1_score_temp)

# Print the results
print("Temporal Set - Stratified Time Series Cross-Validation F1 Scores:", temporal_cv_scores)
print("Temporal Set - Mean F1 Score:", np.mean(temporal_cv_scores))

# Optuna optimization for XGBoost hyperparameters
def objective(trial):
    param = {
        'objective': 'binary:logistic',
        'n_estimators': trial.suggest_int('n_estimators', 150, 300),
        'max_depth': trial.suggest_int('max_depth', 7, 20),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.4, 0.8),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.8),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 0.6)
    }
    model = XGBClassifier(**param, random_state=777, eval_metric='logloss')
    cv_scores = cross_val_score(model, x_rand_balanced, y_rand_balanced, cv=kf, scoring='f1')
    return np.mean(cv_scores)

# Optimize hyperparameters for the Random Set
study_rand = optuna.create_study(direction='maximize')
study_rand.optimize(objective, n_trials=50, timeout=600)
print("Best Parameters (Random Set):", study_rand.best_params)
print("Best F1 Score (Random Set):", study_rand.best_value)

# Visualization of Optimization History
plt.figure(figsize=(10, 6))
optuna.visualization.matplotlib.plot_optimization_history(study_rand)
plt.title('Optimization History XGBoost (Random Set)')
plt.show()

# Retrain the model with the best parameters for the Random Set
best_model_rand = XGBClassifier(**study_rand.best_params, random_state=777, eval_metric='logloss')
best_model_rand.fit(x_rand_balanced, y_rand_balanced)

# Evaluate the best model on the validation set
val_rand_predictions = best_model_rand.predict(x_val_rand)
val_rand_accuracy = accuracy_score(y_val_rand, val_rand_predictions)
val_rand_precision = precision_score(y_val_rand, val_rand_predictions)
val_rand_recall = recall_score(y_val_rand, val_rand_predictions)
val_rand_f1 = f1_score(y_val_rand, val_rand_predictions)
val_rand_auc = roc_auc_score(y_val_rand, val_rand_predictions)

print("\nValidation Metrics (Random Set):")
print(f"Accuracy: {val_rand_accuracy}")
print(f"Precision: {val_rand_precision}")
print(f"Recall: {val_rand_recall}")
print(f"F1-Score: {val_rand_f1}")
print(f"AUC-ROC: {val_rand_auc}")

# Confusion Matrix for the Random Set
conf_matrix_rand = confusion_matrix(y_val_rand, val_rand_predictions)
print("Confusion Matrix (Random Set):")
print(conf_matrix_rand)

# Repeat the process for the Temporal Set
def objective_temp(trial):
    param = {
        'objective': 'binary:logistic',
        'n_estimators': trial.suggest_int('n_estimators', 150, 300),
        'max_depth': trial.suggest_int('max_depth', 7, 20),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.4, 0.8),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.8),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 0.6)
    }
    model = XGBClassifier(**param, random_state=777, eval_metric='logloss')
    cv_scores = []
    for train_index, val_index in stratified_splits:
        X_train, X_val = x_temp_balanced.iloc[train_index], x_temp_balanced.iloc[val_index]
        y_train, y_val = y_temp_balanced.iloc[train_index], y_temp_balanced.iloc[val_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        cv_scores.append(f1_score(y_val, y_pred))
    return np.mean(cv_scores)

# Optimize hyperparameters for the Temporal Set
study_temp = optuna.create_study(direction='maximize')
study_temp.optimize(objective_temp, n_trials=50, timeout=600)
print("Best Parameters (Temporal Set):", study_temp.best_params)
print("Best F1 Score (Temporal Set):", study_temp.best_value)

# Visualization of Optimization History for Temporal Set
plt.figure(figsize=(10, 6))
optuna.visualization.matplotlib.plot_optimization_history(study_temp)
plt.title('Optimization History XGBoost (temporal Set)')
plt.show()

# Retrain the model with the best parameters for the Temporal Set
best_model_temp = XGBClassifier(**study_temp.best_params, random_state=777, eval_metric='logloss')
best_model_temp.fit(x_temp_balanced, y_temp_balanced)

# Evaluate the best model on the validation set for the Temporal Set
val_temp_predictions = best_model_temp.predict(x_val_temp)
val_temp_accuracy = accuracy_score(y_val_temp, val_temp_predictions)
val_temp_precision = precision_score(y_val_temp, val_temp_predictions)
val_temp_recall = recall_score(y_val_temp, val_temp_predictions)
val_temp_f1 = f1_score(y_val_temp, val_temp_predictions)
val_temp_auc = roc_auc_score(y_val_temp, val_temp_predictions)

print("\nValidation Metrics (Temporal Set):")
print(f"Accuracy: {val_temp_accuracy}")
print(f"Precision: {val_temp_precision}")
print(f"Recall: {val_temp_recall}")
print(f"F1-Score: {val_temp_f1}")
print(f"AUC-ROC: {val_temp_auc}")

# Confusion Matrix for the Temporal Set
conf_matrix_temp = confusion_matrix(y_val_temp, val_temp_predictions)
print("Confusion Matrix (Temporal Set):")
print(conf_matrix_temp)

# Combine training and validation datasets for Random Set
X_train_val_rand = pd.concat([x_rand_balanced, x_val_rand], axis=0)
y_train_val_rand = pd.concat([y_rand_balanced, y_val_rand], axis=0)

# Combine training and validation datasets for Temporal Set
X_train_val_temp = pd.concat([x_temp_balanced, x_val_temp], axis=0)
y_train_val_temp = pd.concat([y_temp_balanced, y_val_temp], axis=0)

# Fit final XGBoost model on the combined dataset for the Random Set
final_model_rand_xgb = XGBClassifier(**study_rand.best_params, random_state=777, eval_metric='logloss')
final_model_rand_xgb.fit(X_train_val_rand, y_train_val_rand)

# Fit final XGBoost model on the combined dataset for the Temporal Set
final_model_temp_xgb = XGBClassifier(**study_temp.best_params, random_state=777, eval_metric='logloss')
final_model_temp_xgb.fit(X_train_val_temp, y_train_val_temp)

# Test Set Evaluation for the Random Set
test_rand_predictions_xgb = final_model_rand_xgb.predict(x_test_rand)
test_rand_f1_xgb = f1_score(y_test_rand, test_rand_predictions_xgb)
test_rand_accuracy_xgb = accuracy_score(y_test_rand, test_rand_predictions_xgb)
test_rand_precision_xgb = precision_score(y_test_rand, test_rand_predictions_xgb)
test_rand_recall_xgb = recall_score(y_test_rand, test_rand_predictions_xgb)
test_rand_roc_auc_xgb = roc_auc_score(y_test_rand, final_model_rand_xgb.predict_proba(x_test_rand)[:, 1])
test_rand_conf_matrix_xgb = confusion_matrix(y_test_rand, test_rand_predictions_xgb)

print("\nTest Metrics (Random Set - XGBoost):")
print(f"Accuracy: {test_rand_accuracy_xgb}")
print(f"Precision: {test_rand_precision_xgb}")
print(f"Recall: {test_rand_recall_xgb}")
print(f"F1-Score: {test_rand_f1_xgb}")
print(f"AUC-ROC: {test_rand_roc_auc_xgb}")
print(f"Confusion Matrix:\n{test_rand_conf_matrix_xgb}")

# Test Set Evaluation for the Temporal Set
test_temp_predictions_xgb = final_model_temp_xgb.predict(x_test_temp)
test_temp_f1_xgb = f1_score(y_test_temp, test_temp_predictions_xgb)
test_temp_accuracy_xgb = accuracy_score(y_test_temp, test_temp_predictions_xgb)
test_temp_precision_xgb = precision_score(y_test_temp, test_temp_predictions_xgb)
test_temp_recall_xgb = recall_score(y_test_temp, test_temp_predictions_xgb)
test_temp_roc_auc_xgb = roc_auc_score(y_test_temp, final_model_temp_xgb.predict_proba(x_test_temp)[:, 1])
test_temp_conf_matrix_xgb = confusion_matrix(y_test_temp, test_temp_predictions_xgb)

print("\nTest Metrics (Temporal Set - XGBoost):")
print(f"Accuracy: {test_temp_accuracy_xgb}")
print(f"Precision: {test_temp_precision_xgb}")
print(f"Recall: {test_temp_recall_xgb}")
print(f"F1-Score: {test_temp_f1_xgb}")
print(f"AUC-ROC: {test_temp_roc_auc_xgb}")
print(f"Confusion Matrix:\n{test_temp_conf_matrix_xgb}")

# PERMUTATION IMPORTANCE

# Permutation Importance for the Random Set
perm_importance_rand_xgb = permutation_importance(final_model_rand_xgb, x_val_rand, y_val_rand, n_repeats=10, random_state=777)
importance_rand_df_xgb = pd.DataFrame({
    'Feature': x_rand_balanced.columns,
    'Importance': perm_importance_rand_xgb.importances_mean
})

# Filter for better visualization
importance_rand_df_xgb = importance_rand_df_xgb[(importance_rand_df_xgb['Importance'] > 0.001) | (importance_rand_df_xgb['Importance'] < -0.001)]
importance_rand_df_xgb.sort_values(by='Importance', ascending=False, inplace=True)

# Plot Permutation Importance for the Random Set
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_rand_df_xgb)
plt.title('Permutation Importance (Random Set) - XGBoost')
plt.show()

# Permutation Importance for the Temporal Set
perm_importance_temp_xgb = permutation_importance(final_model_temp_xgb, x_val_temp, y_val_temp, n_repeats=10, random_state=777)
importance_temp_df_xgb = pd.DataFrame({
    'Feature': x_temp_balanced.columns,
    'Importance': perm_importance_temp_xgb.importances_mean
})

# Filter for better visualization
importance_temp_df_xgb = importance_temp_df_xgb[(importance_temp_df_xgb['Importance'] > 0.001) | (importance_temp_df_xgb['Importance'] < -0.001)]
importance_temp_df_xgb.sort_values(by='Importance', ascending=False, inplace=True)

# Plot Permutation Importance for the Temporal Set
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_temp_df_xgb)
plt.title('Permutation Importance (Temporal Set) - XGBoost')
plt.show()

""" RESULTS """