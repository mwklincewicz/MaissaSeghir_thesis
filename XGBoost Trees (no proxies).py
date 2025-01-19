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

""" RESULTS
Random Set - 5-Fold Cross-Validation F1 Scores: [0.56623932 0.55821206 0.56442358 0.5397667  0.57202288]
Random Set - Mean F1 Score: 0.5601329071779085
Temporal Set - Stratified Time Series Cross-Validation F1 Scores: [0.5225933202357564, 0.5663048607833885, 0.6244417377182298, 0.6321548821548821, 0.5734072022160664]
Temporal Set - Mean F1 Score: 0.5837804006216647
[I 2025-01-19 17:17:54,457] A new study created in memory with name: no-name-4fc15fcc-2445-4256-be0d-1006b07d7df2
[I 2025-01-19 17:18:06,029] Trial 0 finished with value: 0.583924154438942 and parameters: {'n_estimators': 215, 'max_depth': 11, 'learning_rate': 0.011150161185383622, 'subsample': 0.7742912124558307, 'colsample_bytree': 0.5300077005175803, 'min_child_weight': 3, 'gamma': 0.5630179246124637}. Best is trial 0 with value: 0.583924154438942.
[I 2025-01-19 17:18:16,136] Trial 1 finished with value: 0.5473321926712454 and parameters: {'n_estimators': 298, 'max_depth': 12, 'learning_rate': 0.18542238889401372, 'subsample': 0.6056170111391163, 'colsample_bytree': 0.5120227383435562, 'min_child_weight': 10, 'gamma': 0.4852218039570256}. Best is trial 0 with value: 0.583924154438942.
[I 2025-01-19 17:18:23,997] Trial 2 finished with value: 0.5396376421374544 and parameters: {'n_estimators': 207, 'max_depth': 9, 'learning_rate': 0.20808858014888695, 'subsample': 0.7516845074416232, 'colsample_bytree': 0.5644698643208262, 'min_child_weight': 3, 'gamma': 0.2019467071288932}. Best is trial 0 with value: 0.583924154438942.
[I 2025-01-19 17:18:30,713] Trial 3 finished with value: 0.5550853225247944 and parameters: {'n_estimators': 240, 'max_depth': 7, 'learning_rate': 0.12913517366460947, 'subsample': 0.6849868464896571, 'colsample_bytree': 0.4463108486338644, 'min_child_weight': 3, 'gamma': 0.26560319450130715}. Best is trial 0 with value: 0.583924154438942.
[I 2025-01-19 17:18:52,115] Trial 4 finished with value: 0.5427041949771538 and parameters: {'n_estimators': 269, 'max_depth': 13, 'learning_rate': 0.12685102259326275, 'subsample': 0.7964320388203276, 'colsample_bytree': 0.7666775513182783, 'min_child_weight': 1, 'gamma': 0.03943289626757522}. Best is trial 0 with value: 0.583924154438942.
[I 2025-01-19 17:19:01,270] Trial 5 finished with value: 0.5442615218482877 and parameters: {'n_estimators': 187, 'max_depth': 19, 'learning_rate': 0.20801604820540437, 'subsample': 0.769815472844049, 'colsample_bytree': 0.4692427237694489, 'min_child_weight': 8, 'gamma': 0.1684079873005749}. Best is trial 0 with value: 0.583924154438942.
[I 2025-01-19 17:19:16,457] Trial 6 finished with value: 0.5459475762316363 and parameters: {'n_estimators': 283, 'max_depth': 13, 'learning_rate': 0.15364646101483728, 'subsample': 0.7750740272920142, 'colsample_bytree': 0.7845621696577947, 'min_child_weight': 1, 'gamma': 0.3680641277578224}. Best is trial 0 with value: 0.583924154438942.
[I 2025-01-19 17:19:25,336] Trial 7 finished with value: 0.5541539920757905 and parameters: {'n_estimators': 221, 'max_depth': 12, 'learning_rate': 0.10126298572915225, 'subsample': 0.6946690314414461, 'colsample_bytree': 0.6913328944896737, 'min_child_weight': 10, 'gamma': 0.1324849455780566}. Best is trial 0 with value: 0.583924154438942.
[I 2025-01-19 17:19:36,039] Trial 8 finished with value: 0.5466387898796542 and parameters: {'n_estimators': 266, 'max_depth': 19, 'learning_rate': 0.20198029748934365, 'subsample': 0.7125164009767347, 'colsample_bytree': 0.5005293685572707, 'min_child_weight': 10, 'gamma': 0.2526147788865129}. Best is trial 0 with value: 0.583924154438942.
[I 2025-01-19 17:19:42,342] Trial 9 finished with value: 0.5521131949020192 and parameters: {'n_estimators': 180, 'max_depth': 9, 'learning_rate': 0.1870377876988769, 'subsample': 0.5226934318721363, 'colsample_bytree': 0.680670326306097, 'min_child_weight': 7, 'gamma': 0.34720695123830636}. Best is trial 0 with value: 0.583924154438942.
[I 2025-01-19 17:19:51,404] Trial 10 finished with value: 0.5726452611256613 and parameters: {'n_estimators': 158, 'max_depth': 16, 'learning_rate': 0.03943629389053767, 'subsample': 0.41948557635712935, 'colsample_bytree': 0.610269719625983, 'min_child_weight': 4, 'gamma': 0.5716009025545352}. Best is trial 0 with value: 0.583924154438942.
[I 2025-01-19 17:20:01,441] Trial 11 finished with value: 0.5768978035823215 and parameters: {'n_estimators': 150, 'max_depth': 17, 'learning_rate': 0.0104194725582956, 'subsample': 0.45311647074983885, 'colsample_bytree': 0.6080357502554535, 'min_child_weight': 4, 'gamma': 0.5823970859792669}. Best is trial 0 with value: 0.583924154438942.
[I 2025-01-19 17:20:09,638] Trial 12 finished with value: 0.5835351401296774 and parameters: {'n_estimators': 152, 'max_depth': 16, 'learning_rate': 0.012901860921032349, 'subsample': 0.4070431262591926, 'colsample_bytree': 0.590227526732124, 'min_child_weight': 5, 'gamma': 0.5912552229040005}. Best is trial 0 with value: 0.583924154438942.
[I 2025-01-19 17:20:21,403] Trial 13 finished with value: 0.5606316034405299 and parameters: {'n_estimators': 236, 'max_depth': 16, 'learning_rate': 0.06623387183526694, 'subsample': 0.5888624221797893, 'colsample_bytree': 0.5557511919072001, 'min_child_weight': 6, 'gamma': 0.45011644455128397}. Best is trial 0 with value: 0.583924154438942.
[I 2025-01-19 17:20:28,698] Trial 14 finished with value: 0.5443756400635744 and parameters: {'n_estimators': 184, 'max_depth': 10, 'learning_rate': 0.2905805920390012, 'subsample': 0.5012500788018183, 'colsample_bytree': 0.4125255315942234, 'min_child_weight': 5, 'gamma': 0.48147336550251063}. Best is trial 0 with value: 0.583924154438942.
[I 2025-01-19 17:20:46,633] Trial 15 finished with value: 0.5758181000455564 and parameters: {'n_estimators': 207, 'max_depth': 15, 'learning_rate': 0.012251688489594844, 'subsample': 0.6203638502895154, 'colsample_bytree': 0.6412264145428516, 'min_child_weight': 2, 'gamma': 0.599859555863414}. Best is trial 0 with value: 0.583924154438942.
[I 2025-01-19 17:20:54,040] Trial 16 finished with value: 0.5601341337996555 and parameters: {'n_estimators': 169, 'max_depth': 11, 'learning_rate': 0.0733266121673951, 'subsample': 0.5329342778185855, 'colsample_bytree': 0.5366141254753789, 'min_child_weight': 5, 'gamma': 0.38679793231585424}. Best is trial 0 with value: 0.583924154438942.
[I 2025-01-19 17:21:04,583] Trial 17 finished with value: 0.5602750145454658 and parameters: {'n_estimators': 249, 'max_depth': 14, 'learning_rate': 0.057485234424910436, 'subsample': 0.40198425212566025, 'colsample_bytree': 0.6668288412506446, 'min_child_weight': 7, 'gamma': 0.528123475246483}. Best is trial 0 with value: 0.583924154438942.
[I 2025-01-19 17:21:19,501] Trial 18 finished with value: 0.5466893456767421 and parameters: {'n_estimators': 204, 'max_depth': 18, 'learning_rate': 0.09359450084134698, 'subsample': 0.6416250604904428, 'colsample_bytree': 0.7395014231323891, 'min_child_weight': 3, 'gamma': 0.42544085462709313}. Best is trial 0 with value: 0.583924154438942.
[I 2025-01-19 17:21:29,395] Trial 19 finished with value: 0.539734464166855 and parameters: {'n_estimators': 220, 'max_depth': 15, 'learning_rate': 0.2702015364001203, 'subsample': 0.5548198360612764, 'colsample_bytree': 0.5775673675845233, 'min_child_weight': 6, 'gamma': 0.528502823364703}. Best is trial 0 with value: 0.583924154438942.
[I 2025-01-19 17:21:35,127] Trial 20 finished with value: 0.5450782310392286 and parameters: {'n_estimators': 196, 'max_depth': 7, 'learning_rate': 0.24801005348224928, 'subsample': 0.6633457740859707, 'colsample_bytree': 0.4918531810561193, 'min_child_weight': 4, 'gamma': 0.5256285432807281}. Best is trial 0 with value: 0.583924154438942.
[I 2025-01-19 17:21:48,007] Trial 21 finished with value: 0.5752727486454923 and parameters: {'n_estimators': 150, 'max_depth': 17, 'learning_rate': 0.015801541206685088, 'subsample': 0.4467026876553015, 'colsample_bytree': 0.6157469970591132, 'min_child_weight': 2, 'gamma': 0.5827253465316043}. Best is trial 0 with value: 0.583924154438942.
[I 2025-01-19 17:21:58,427] Trial 22 finished with value: 0.5678273134941924 and parameters: {'n_estimators': 167, 'max_depth': 17, 'learning_rate': 0.04089505914103718, 'subsample': 0.4721034169459454, 'colsample_bytree': 0.6044150994333788, 'min_child_weight': 4, 'gamma': 0.5313962217280159}. Best is trial 0 with value: 0.583924154438942.
[I 2025-01-19 17:22:07,195] Trial 23 finished with value: 0.5785154354582682 and parameters: {'n_estimators': 152, 'max_depth': 20, 'learning_rate': 0.01115349148364492, 'subsample': 0.44745101173675544, 'colsample_bytree': 0.5326328923447238, 'min_child_weight': 5, 'gamma': 0.5915170464495051}. Best is trial 0 with value: 0.583924154438942.
[I 2025-01-19 17:22:17,074] Trial 24 finished with value: 0.5726673020496652 and parameters: {'n_estimators': 169, 'max_depth': 20, 'learning_rate': 0.039690239563972426, 'subsample': 0.489960761724563, 'colsample_bytree': 0.5530904201360697, 'min_child_weight': 5, 'gamma': 0.4288669128382363}. Best is trial 0 with value: 0.583924154438942.
[I 2025-01-19 17:22:25,320] Trial 25 finished with value: 0.5694813525687781 and parameters: {'n_estimators': 177, 'max_depth': 20, 'learning_rate': 0.03164341810010132, 'subsample': 0.43698950475677173, 'colsample_bytree': 0.4479450450131638, 'min_child_weight': 7, 'gamma': 0.48524008948815245}. Best is trial 0 with value: 0.583924154438942.
[I 2025-01-19 17:22:32,643] Trial 26 finished with value: 0.5600915297187431 and parameters: {'n_estimators': 163, 'max_depth': 14, 'learning_rate': 0.08802969094994943, 'subsample': 0.40335673135419786, 'colsample_bytree': 0.5239163185382886, 'min_child_weight': 6, 'gamma': 0.5503730435287544}. Best is trial 0 with value: 0.583924154438942.
[I 2025-01-19 17:22:40,173] Trial 27 finished with value: 0.5688067083268284 and parameters: {'n_estimators': 191, 'max_depth': 11, 'learning_rate': 0.0544811203369831, 'subsample': 0.5694847374082294, 'colsample_bytree': 0.5805048667339651, 'min_child_weight': 8, 'gamma': 0.33008451749321044}. Best is trial 0 with value: 0.583924154438942.
[I 2025-01-19 17:23:05,107] Trial 28 finished with value: 0.5560770060171188 and parameters: {'n_estimators': 261, 'max_depth': 19, 'learning_rate': 0.02453811200769563, 'subsample': 0.7250460016263791, 'colsample_bytree': 0.6484764144061048, 'min_child_weight': 2, 'gamma': 0.4820428641747161}. Best is trial 0 with value: 0.583924154438942.
[I 2025-01-19 17:23:16,677] Trial 29 finished with value: 0.5541713440613641 and parameters: {'n_estimators': 292, 'max_depth': 12, 'learning_rate': 0.07842092974541161, 'subsample': 0.4759732041393136, 'colsample_bytree': 0.5229749396336217, 'min_child_weight': 5, 'gamma': 0.4938687990798173}. Best is trial 0 with value: 0.583924154438942.
[I 2025-01-19 17:23:23,830] Trial 30 finished with value: 0.550003320750909 and parameters: {'n_estimators': 175, 'max_depth': 9, 'learning_rate': 0.11388338767218124, 'subsample': 0.5140191322089802, 'colsample_bytree': 0.7254842387188869, 'min_child_weight': 3, 'gamma': 0.03929232702784874}. Best is trial 0 with value: 0.583924154438942.
[I 2025-01-19 17:23:33,383] Trial 31 finished with value: 0.5761655037078635 and parameters: {'n_estimators': 150, 'max_depth': 17, 'learning_rate': 0.015013551908556392, 'subsample': 0.44852741153365516, 'colsample_bytree': 0.6325964578866752, 'min_child_weight': 4, 'gamma': 0.5916195362256095}. Best is trial 0 with value: 0.583924154438942.
[I 2025-01-19 17:23:42,836] Trial 32 finished with value: 0.5629367808886789 and parameters: {'n_estimators': 155, 'max_depth': 18, 'learning_rate': 0.05028823425630301, 'subsample': 0.4627944341742814, 'colsample_bytree': 0.5835901195620214, 'min_child_weight': 4, 'gamma': 0.5541897198560608}. Best is trial 0 with value: 0.583924154438942.
[I 2025-01-19 17:23:52,422] Trial 33 finished with value: 0.5752738084059327 and parameters: {'n_estimators': 158, 'max_depth': 15, 'learning_rate': 0.03004271314728529, 'subsample': 0.42969725962691113, 'colsample_bytree': 0.5417244448333242, 'min_child_weight': 3, 'gamma': 0.5642910742373497}. Best is trial 0 with value: 0.583924154438942.
[I 2025-01-19 17:24:03,782] Trial 34 finished with value: 0.574427177924637 and parameters: {'n_estimators': 200, 'max_depth': 18, 'learning_rate': 0.012712078553704459, 'subsample': 0.5406311915372415, 'colsample_bytree': 0.49750957608157953, 'min_child_weight': 5, 'gamma': 0.5967724265405527}. Best is trial 0 with value: 0.583924154438942.
[I 2025-01-19 17:24:14,488] Trial 35 finished with value: 0.5442445904010651 and parameters: {'n_estimators': 211, 'max_depth': 16, 'learning_rate': 0.1419860061369109, 'subsample': 0.42233745763828606, 'colsample_bytree': 0.4770551762230319, 'min_child_weight': 3, 'gamma': 0.5110908552901837}. Best is trial 0 with value: 0.583924154438942.
[I 2025-01-19 17:24:27,473] Trial 36 finished with value: 0.5579036929888482 and parameters: {'n_estimators': 234, 'max_depth': 20, 'learning_rate': 0.0635533068551576, 'subsample': 0.4597809948468274, 'colsample_bytree': 0.5738376128893796, 'min_child_weight': 4, 'gamma': 0.4358609206407664}. Best is trial 0 with value: 0.583924154438942.
[I 2025-01-19 17:24:38,667] Trial 37 finished with value: 0.5803078876278758 and parameters: {'n_estimators': 162, 'max_depth': 13, 'learning_rate': 0.01021006923594958, 'subsample': 0.49839371838048474, 'colsample_bytree': 0.5162897198785409, 'min_child_weight': 2, 'gamma': 0.39585756138713524}. Best is trial 0 with value: 0.583924154438942.
[I 2025-01-19 17:24:48,407] Trial 38 finished with value: 0.5358758748867493 and parameters: {'n_estimators': 162, 'max_depth': 13, 'learning_rate': 0.2350513619999665, 'subsample': 0.48929188918919014, 'colsample_bytree': 0.45597253002440225, 'min_child_weight': 1, 'gamma': 0.2876759584356132}. Best is trial 0 with value: 0.583924154438942.
[I 2025-01-19 17:24:57,891] Trial 39 finished with value: 0.5540992694380141 and parameters: {'n_estimators': 191, 'max_depth': 12, 'learning_rate': 0.10993561728163342, 'subsample': 0.7404276575327523, 'colsample_bytree': 0.40835493707415405, 'min_child_weight': 2, 'gamma': 0.3893732440346991}. Best is trial 0 with value: 0.583924154438942.
[I 2025-01-19 17:25:06,482] Trial 40 finished with value: 0.5447976741675118 and parameters: {'n_estimators': 174, 'max_depth': 10, 'learning_rate': 0.17530543574020144, 'subsample': 0.6071390633770137, 'colsample_bytree': 0.5168554002478597, 'min_child_weight': 1, 'gamma': 0.1650623248651698}. Best is trial 0 with value: 0.583924154438942.
[I 2025-01-19 17:25:15,544] Trial 41 finished with value: 0.5751874255650328 and parameters: {'n_estimators': 151, 'max_depth': 14, 'learning_rate': 0.025614940999879498, 'subsample': 0.4127177775100398, 'colsample_bytree': 0.5936012127079296, 'min_child_weight': 3, 'gamma': 0.553507155853747}. Best is trial 0 with value: 0.583924154438942.
[I 2025-01-19 17:25:25,425] Trial 42 finished with value: 0.5645734527359625 and parameters: {'n_estimators': 159, 'max_depth': 13, 'learning_rate': 0.045031276405241114, 'subsample': 0.4411918534249029, 'colsample_bytree': 0.5485039579188968, 'min_child_weight': 2, 'gamma': 0.23073376831811288}. Best is trial 0 with value: 0.583924154438942.
[I 2025-01-19 17:25:32,892] Trial 43 finished with value: 0.5748480298807002 and parameters: {'n_estimators': 166, 'max_depth': 11, 'learning_rate': 0.02899032516605196, 'subsample': 0.5016121900462516, 'colsample_bytree': 0.47488677779128224, 'min_child_weight': 4, 'gamma': 0.0838526042207284}. Best is trial 0 with value: 0.583924154438942.
[I 2025-01-19 17:25:39,217] Trial 44 finished with value: 0.5943656508451192 and parameters: {'n_estimators': 183, 'max_depth': 8, 'learning_rate': 0.012861041706125408, 'subsample': 0.6767997969777552, 'colsample_bytree': 0.5599137367354491, 'min_child_weight': 6, 'gamma': 0.5670670746627068}. Best is trial 44 with value: 0.5943656508451192.
[I 2025-01-19 17:25:45,471] Trial 45 finished with value: 0.5715787513676778 and parameters: {'n_estimators': 181, 'max_depth': 9, 'learning_rate': 0.047911827308958446, 'subsample': 0.7790444745026343, 'colsample_bytree': 0.5064354476165323, 'min_child_weight': 6, 'gamma': 0.49909037587158195}. Best is trial 44 with value: 0.5943656508451192.
[I 2025-01-19 17:25:52,065] Trial 46 finished with value: 0.5796827257170708 and parameters: {'n_estimators': 216, 'max_depth': 8, 'learning_rate': 0.02297940273618898, 'subsample': 0.6712993657242685, 'colsample_bytree': 0.5613548473837139, 'min_child_weight': 9, 'gamma': 0.5587444899879209}. Best is trial 44 with value: 0.5943656508451192.
[I 2025-01-19 17:25:58,601] Trial 47 finished with value: 0.5599279510611656 and parameters: {'n_estimators': 215, 'max_depth': 8, 'learning_rate': 0.07898004631741747, 'subsample': 0.6958869664281734, 'colsample_bytree': 0.5637725277949861, 'min_child_weight': 9, 'gamma': 0.4587581322328623}. Best is trial 44 with value: 0.5943656508451192.
[I 2025-01-19 17:26:05,493] Trial 48 finished with value: 0.580072527245864 and parameters: {'n_estimators': 226, 'max_depth': 8, 'learning_rate': 0.024907992654688536, 'subsample': 0.6559032218567009, 'colsample_bytree': 0.5610495459132777, 'min_child_weight': 9, 'gamma': 0.3957077041267717}. Best is trial 44 with value: 0.5943656508451192.
[I 2025-01-19 17:26:11,722] Trial 49 finished with value: 0.5733552645069654 and parameters: {'n_estimators': 246, 'max_depth': 7, 'learning_rate': 0.06111699783471718, 'subsample': 0.6349625017909906, 'colsample_bytree': 0.4349131013551646, 'min_child_weight': 8, 'gamma': 0.3417588560915842}. Best is trial 44 with value: 0.5943656508451192.
Best Parameters (Random Set): {'n_estimators': 183, 'max_depth': 8, 'learning_rate': 0.012861041706125408, 'subsample': 0.6767997969777552, 'colsample_bytree': 0.5599137367354491, 'min_child_weight': 6, 'gamma': 0.5670670746627068}
Best F1 Score (Random Set): 0.5943656508451192
c:\Users\msegh\MaissaSeghir_thesis\XGBoost Trees (no proxies).py:128: ExperimentalWarning: plot_optimization_history is experimental (supported from v2.2.0). The interface can change in the future.
  optuna.visualization.matplotlib.plot_optimization_history(study_rand)

Validation Metrics (Random Set):
Accuracy: 0.5986881335718545
Precision: 0.5749086479902558
Recall: 0.592964824120603
F1-Score: 0.5837971552257266
AUC-ROC: 0.5984120374859543
Confusion Matrix (Random Set):
[[1064  698]
 [ 648  944]]
[I 2025-01-19 17:26:23,826] A new study created in memory with name: no-name-92fc188d-c2ef-48aa-9ceb-f20477494857
[I 2025-01-19 17:26:36,457] Trial 0 finished with value: 0.592777257762419 and parameters: {'n_estimators': 255, 'max_depth': 14, 'learning_rate': 0.062383302400109725, 'subsample': 0.6996378353414847, 'colsample_bytree': 0.6096601547629715, 'min_child_weight': 5, 'gamma': 0.2077058078359486}. Best is trial 0 with value: 0.592777257762419.
[I 2025-01-19 17:26:42,652] Trial 1 finished with value: 0.5786174273558795 and parameters: {'n_estimators': 157, 'max_depth': 12, 'learning_rate': 0.27200871715358965, 'subsample': 0.7026371187091027, 'colsample_bytree': 0.4225578761286121, 'min_child_weight': 7, 'gamma': 0.4353034734228119}. Best is trial 0 with value: 0.592777257762419.
[I 2025-01-19 17:26:48,085] Trial 2 finished with value: 0.5932641914012038 and parameters: {'n_estimators': 195, 'max_depth': 7, 'learning_rate': 0.11634639851903532, 'subsample': 0.6936777481032184, 'colsample_bytree': 0.6163261026221898, 'min_child_weight': 7, 'gamma': 0.22981366641121376}. Best is trial 2 with value: 0.5932641914012038.
[I 2025-01-19 17:26:58,035] Trial 3 finished with value: 0.5721992910821154 and parameters: {'n_estimators': 270, 'max_depth': 12, 'learning_rate': 0.29260948378552554, 'subsample': 0.5913662386991029, 'colsample_bytree': 0.47883782340627756, 'min_child_weight': 7, 'gamma': 0.05521658059077648}. Best is trial 2 with value: 0.5932641914012038.
[I 2025-01-19 17:27:06,342] Trial 4 finished with value: 0.5801644888328993 and parameters: {'n_estimators': 172, 'max_depth': 15, 'learning_rate': 0.21520429310983316, 'subsample': 0.716467689166604, 'colsample_bytree': 0.6841856358862566, 'min_child_weight': 8, 'gamma': 0.3525720734338165}. Best is trial 2 with value: 0.5932641914012038.
[I 2025-01-19 17:27:11,391] Trial 5 finished with value: 0.599113817012354 and parameters: {'n_estimators': 177, 'max_depth': 7, 'learning_rate': 0.06610262932900621, 'subsample': 0.7105402971402498, 'colsample_bytree': 0.4087529796840524, 'min_child_weight': 2, 'gamma': 0.36763243299890086}. Best is trial 5 with value: 0.599113817012354.
[I 2025-01-19 17:27:18,674] Trial 6 finished with value: 0.6159843524529189 and parameters: {'n_estimators': 194, 'max_depth': 15, 'learning_rate': 0.015852184825474812, 'subsample': 0.4075224970370343, 'colsample_bytree': 0.4063530484489446, 'min_child_weight': 10, 'gamma': 0.2145088218891674}. Best is trial 6 with value: 0.6159843524529189.
[I 2025-01-19 17:27:29,206] Trial 7 finished with value: 0.5790301596228467 and parameters: {'n_estimators': 157, 'max_depth': 11, 'learning_rate': 0.1997084622480965, 'subsample': 0.6802166287054294, 'colsample_bytree': 0.787551046636067, 'min_child_weight': 1, 'gamma': 0.09656797419467193}. Best is trial 6 with value: 0.6159843524529189.
[I 2025-01-19 17:27:41,943] Trial 8 finished with value: 0.5894926150343499 and parameters: {'n_estimators': 241, 'max_depth': 11, 'learning_rate': 0.0825744731050341, 'subsample': 0.6242142144138203, 'colsample_bytree': 0.7877255439089945, 'min_child_weight': 2, 'gamma': 0.4362771781176623}. Best is trial 6 with value: 0.6159843524529189.
[I 2025-01-19 17:27:50,441] Trial 9 finished with value: 0.5818573113161317 and parameters: {'n_estimators': 290, 'max_depth': 8, 'learning_rate': 0.2705448387201785, 'subsample': 0.7908711559367831, 'colsample_bytree': 0.6602134058191884, 'min_child_weight': 6, 'gamma': 0.5343485207801215}. Best is trial 6 with value: 0.6159843524529189.
[I 2025-01-19 17:27:58,516] Trial 10 finished with value: 0.6034363604536864 and parameters: {'n_estimators': 209, 'max_depth': 19, 'learning_rate': 0.027244403998584184, 'subsample': 0.40829280852617333, 'colsample_bytree': 0.5201152900392977, 'min_child_weight': 10, 'gamma': 0.16336610487498732}. Best is trial 6 with value: 0.6159843524529189.
[I 2025-01-19 17:28:06,677] Trial 11 finished with value: 0.6177960139547108 and parameters: {'n_estimators': 210, 'max_depth': 20, 'learning_rate': 0.013011560451343829, 'subsample': 0.40149498366787795, 'colsample_bytree': 0.5134903369572367, 'min_child_weight': 10, 'gamma': 0.15273506792983949}. Best is trial 11 with value: 0.6177960139547108.
[I 2025-01-19 17:28:15,045] Trial 12 finished with value: 0.6206629203876773 and parameters: {'n_estimators': 214, 'max_depth': 20, 'learning_rate': 0.010542064668005875, 'subsample': 0.41853841690640564, 'colsample_bytree': 0.5214242763107151, 'min_child_weight': 10, 'gamma': 0.011709020859556996}. Best is trial 12 with value: 0.6206629203876773.
[I 2025-01-19 17:28:24,301] Trial 13 finished with value: 0.5813545333994219 and parameters: {'n_estimators': 228, 'max_depth': 20, 'learning_rate': 0.13925439348552207, 'subsample': 0.4852057067011185, 'colsample_bytree': 0.5330488417121739, 'min_child_weight': 9, 'gamma': 0.0006471467567506217}. Best is trial 12 with value: 0.6206629203876773.
[I 2025-01-19 17:28:37,455] Trial 14 finished with value: 0.6083798379670828 and parameters: {'n_estimators': 216, 'max_depth': 18, 'learning_rate': 0.015587503110035861, 'subsample': 0.48879494122754497, 'colsample_bytree': 0.5383730198099801, 'min_child_weight': 4, 'gamma': 0.10618604469134388}. Best is trial 12 with value: 0.6206629203876773.
[I 2025-01-19 17:28:46,700] Trial 15 finished with value: 0.5929480536718025 and parameters: {'n_estimators': 234, 'max_depth': 17, 'learning_rate': 0.10075218903752187, 'subsample': 0.4818292629401112, 'colsample_bytree': 0.4700339636694846, 'min_child_weight': 9, 'gamma': 0.0074074312024795886}. Best is trial 12 with value: 0.6206629203876773.
[I 2025-01-19 17:28:55,262] Trial 16 finished with value: 0.5925028594788894 and parameters: {'n_estimators': 202, 'max_depth': 17, 'learning_rate': 0.04482080826283971, 'subsample': 0.5294732240243893, 'colsample_bytree': 0.5678348970075973, 'min_child_weight': 10, 'gamma': 0.11725363918651972}. Best is trial 12 with value: 0.6206629203876773.
[I 2025-01-19 17:29:04,888] Trial 17 finished with value: 0.5696855827780601 and parameters: {'n_estimators': 259, 'max_depth': 20, 'learning_rate': 0.18273262967265225, 'subsample': 0.442943453666635, 'colsample_bytree': 0.49121852661850635, 'min_child_weight': 9, 'gamma': 0.2989356109465656}. Best is trial 12 with value: 0.6206629203876773.
[I 2025-01-19 17:29:13,350] Trial 18 finished with value: 0.5833210499797347 and parameters: {'n_estimators': 184, 'max_depth': 17, 'learning_rate': 0.154834490921015, 'subsample': 0.5711992203598345, 'colsample_bytree': 0.569295502115932, 'min_child_weight': 8, 'gamma': 0.05742407113152136}. Best is trial 12 with value: 0.6206629203876773.
[I 2025-01-19 17:29:25,298] Trial 19 finished with value: 0.5964637165235974 and parameters: {'n_estimators': 218, 'max_depth': 19, 'learning_rate': 0.04470455116676333, 'subsample': 0.44625224444868933, 'colsample_bytree': 0.45315037347200826, 'min_child_weight': 4, 'gamma': 0.28236851855937534}. Best is trial 12 with value: 0.6206629203876773.
[I 2025-01-19 17:29:36,167] Trial 20 finished with value: 0.5793658269783217 and parameters: {'n_estimators': 244, 'max_depth': 16, 'learning_rate': 0.11511660357771203, 'subsample': 0.5339832340090532, 'colsample_bytree': 0.6923382836172879, 'min_child_weight': 8, 'gamma': 0.15427733385394282}. Best is trial 12 with value: 0.6206629203876773.
[I 2025-01-19 17:29:43,628] Trial 21 finished with value: 0.6238192624060775 and parameters: {'n_estimators': 191, 'max_depth': 20, 'learning_rate': 0.011641931703619435, 'subsample': 0.4123106557646223, 'colsample_bytree': 0.4414373067794537, 'min_child_weight': 10, 'gamma': 0.2080243124600631}. Best is trial 21 with value: 0.6238192624060775.
[I 2025-01-19 17:29:51,296] Trial 22 finished with value: 0.6227368449555856 and parameters: {'n_estimators': 188, 'max_depth': 20, 'learning_rate': 0.011996998318156737, 'subsample': 0.4440462449926071, 'colsample_bytree': 0.5113626847068447, 'min_child_weight': 10, 'gamma': 0.18038682766413802}. Best is trial 21 with value: 0.6238192624060775.
[I 2025-01-19 17:29:58,853] Trial 23 finished with value: 0.6055809594983698 and parameters: {'n_estimators': 188, 'max_depth': 18, 'learning_rate': 0.0442200311850813, 'subsample': 0.43922177242412663, 'colsample_bytree': 0.4424520204900981, 'min_child_weight': 9, 'gamma': 0.2458151873688007}. Best is trial 21 with value: 0.6238192624060775.
[I 2025-01-19 17:30:06,075] Trial 24 finished with value: 0.587390852439991 and parameters: {'n_estimators': 163, 'max_depth': 19, 'learning_rate': 0.07571615694660593, 'subsample': 0.511899212455093, 'colsample_bytree': 0.5694806126037394, 'min_child_weight': 10, 'gamma': 0.34685180662050963}. Best is trial 21 with value: 0.6238192624060775.
[I 2025-01-19 17:30:13,400] Trial 25 finished with value: 0.5937936055309221 and parameters: {'n_estimators': 173, 'max_depth': 18, 'learning_rate': 0.04654415386608589, 'subsample': 0.45589711418851897, 'colsample_bytree': 0.5032652392397515, 'min_child_weight': 9, 'gamma': 0.05797403402063922}. Best is trial 21 with value: 0.6238192624060775.
[I 2025-01-19 17:30:21,794] Trial 26 finished with value: 0.5993445341318745 and parameters: {'n_estimators': 201, 'max_depth': 20, 'learning_rate': 0.03806508349026255, 'subsample': 0.42769094295521604, 'colsample_bytree': 0.44028987982012713, 'min_child_weight': 8, 'gamma': 0.1838689007733362}. Best is trial 21 with value: 0.6238192624060775.
[I 2025-01-19 17:30:33,710] Trial 27 finished with value: 0.58110404437099 and parameters: {'n_estimators': 224, 'max_depth': 19, 'learning_rate': 0.08460909313423225, 'subsample': 0.4731010812670084, 'colsample_bytree': 0.5514606622224061, 'min_child_weight': 6, 'gamma': 0.25397475375577105}. Best is trial 21 with value: 0.6238192624060775.
[I 2025-01-19 17:30:42,203] Trial 28 finished with value: 0.6236877626811136 and parameters: {'n_estimators': 177, 'max_depth': 16, 'learning_rate': 0.011619366384744683, 'subsample': 0.5092992914510578, 'colsample_bytree': 0.46859872373876543, 'min_child_weight': 10, 'gamma': 0.5829994972660192}. Best is trial 21 with value: 0.6238192624060775.
[I 2025-01-19 17:30:50,950] Trial 29 finished with value: 0.5875968260474312 and parameters: {'n_estimators': 167, 'max_depth': 14, 'learning_rate': 0.0609689649376983, 'subsample': 0.5361102811584805, 'colsample_bytree': 0.5984776885137288, 'min_child_weight': 5, 'gamma': 0.5616827170578649}. Best is trial 21 with value: 0.6238192624060775.
[I 2025-01-19 17:30:57,682] Trial 30 finished with value: 0.5809605136090831 and parameters: {'n_estimators': 151, 'max_depth': 16, 'learning_rate': 0.10196023948866446, 'subsample': 0.5624996658615002, 'colsample_bytree': 0.4644013970379921, 'min_child_weight': 9, 'gamma': 0.47768310107461465}. Best is trial 21 with value: 0.6238192624060775.
[I 2025-01-19 17:31:05,165] Trial 31 finished with value: 0.6109176750643619 and parameters: {'n_estimators': 183, 'max_depth': 18, 'learning_rate': 0.028082623217853007, 'subsample': 0.46327783992600285, 'colsample_bytree': 0.4911179759941612, 'min_child_weight': 10, 'gamma': 0.5918758600294236}. Best is trial 21 with value: 0.6238192624060775.
[I 2025-01-19 17:31:12,320] Trial 32 finished with value: 0.6264966677245978 and parameters: {'n_estimators': 189, 'max_depth': 13, 'learning_rate': 0.010526222378915213, 'subsample': 0.42496204179679387, 'colsample_bytree': 0.43441702668124493, 'min_child_weight': 10, 'gamma': 0.41175262428544657}. Best is trial 32 with value: 0.6264966677245978.
[I 2025-01-19 17:31:20,423] Trial 33 finished with value: 0.5967388007045668 and parameters: {'n_estimators': 192, 'max_depth': 13, 'learning_rate': 0.057647640826312976, 'subsample': 0.510845347139206, 'colsample_bytree': 0.4341647402320385, 'min_child_weight': 9, 'gamma': 0.5071581753993363}. Best is trial 32 with value: 0.6264966677245978.
[I 2025-01-19 17:31:28,242] Trial 34 finished with value: 0.6137189956384209 and parameters: {'n_estimators': 181, 'max_depth': 13, 'learning_rate': 0.022496687242151356, 'subsample': 0.5040246519627808, 'colsample_bytree': 0.42116283161830864, 'min_child_weight': 7, 'gamma': 0.40572367221940964}. Best is trial 32 with value: 0.6264966677245978.
[I 2025-01-19 17:31:34,647] Trial 35 finished with value: 0.6027574550051418 and parameters: {'n_estimators': 201, 'max_depth': 9, 'learning_rate': 0.03342122391934953, 'subsample': 0.4311564690462474, 'colsample_bytree': 0.4002505689608166, 'min_child_weight': 8, 'gamma': 0.4684108233320383}. Best is trial 32 with value: 0.6264966677245978.
[I 2025-01-19 17:31:41,677] Trial 36 finished with value: 0.5740261134392147 and parameters: {'n_estimators': 166, 'max_depth': 15, 'learning_rate': 0.25392750334322084, 'subsample': 0.4629635618133444, 'colsample_bytree': 0.471819599158231, 'min_child_weight': 7, 'gamma': 0.3266650920619404}. Best is trial 32 with value: 0.6264966677245978.
[I 2025-01-19 17:31:49,001] Trial 37 finished with value: 0.5951898307238175 and parameters: {'n_estimators': 192, 'max_depth': 12, 'learning_rate': 0.06520541379795758, 'subsample': 0.6516446264653746, 'colsample_bytree': 0.42882030417883493, 'min_child_weight': 10, 'gamma': 0.37837898622777644}. Best is trial 32 with value: 0.6264966677245978.
[I 2025-01-19 17:31:55,020] Trial 38 finished with value: 0.6077500933996461 and parameters: {'n_estimators': 175, 'max_depth': 10, 'learning_rate': 0.029766734472216276, 'subsample': 0.40143714813526626, 'colsample_bytree': 0.45790171624637743, 'min_child_weight': 9, 'gamma': 0.20632165407141237}. Best is trial 32 with value: 0.6264966677245978.
[I 2025-01-19 17:32:01,723] Trial 39 finished with value: 0.5993883251502048 and parameters: {'n_estimators': 158, 'max_depth': 14, 'learning_rate': 0.05131169647990206, 'subsample': 0.42800716684276663, 'colsample_bytree': 0.4946042964420731, 'min_child_weight': 8, 'gamma': 0.4101411538329465}. Best is trial 32 with value: 0.6264966677245978.
[I 2025-01-19 17:32:09,844] Trial 40 finished with value: 0.5910457697550571 and parameters: {'n_estimators': 201, 'max_depth': 16, 'learning_rate': 0.07657084228122116, 'subsample': 0.4541014276672315, 'colsample_bytree': 0.6217586457633572, 'min_child_weight': 10, 'gamma': 0.27893670515883207}. Best is trial 32 with value: 0.6264966677245978.
[I 2025-01-19 17:32:18,053] Trial 41 finished with value: 0.6178977791114653 and parameters: {'n_estimators': 209, 'max_depth': 20, 'learning_rate': 0.011467427899532454, 'subsample': 0.4196204178371746, 'colsample_bytree': 0.4784485543410944, 'min_child_weight': 10, 'gamma': 0.19459352436936161}. Best is trial 32 with value: 0.6264966677245978.
[I 2025-01-19 17:32:25,551] Trial 42 finished with value: 0.6144266808994615 and parameters: {'n_estimators': 187, 'max_depth': 19, 'learning_rate': 0.013601685603153104, 'subsample': 0.4185986361617055, 'colsample_bytree': 0.5173909575792931, 'min_child_weight': 10, 'gamma': 0.1292033114458187}. Best is trial 32 with value: 0.6264966677245978.
[I 2025-01-19 17:32:33,095] Trial 43 finished with value: 0.6040034748261421 and parameters: {'n_estimators': 179, 'max_depth': 12, 'learning_rate': 0.02852037724714332, 'subsample': 0.75741725179605, 'colsample_bytree': 0.4537187639766205, 'min_child_weight': 9, 'gamma': 0.075309235766082}. Best is trial 32 with value: 0.6264966677245978.
[I 2025-01-19 17:32:40,519] Trial 44 finished with value: 0.6260295426788092 and parameters: {'n_estimators': 197, 'max_depth': 13, 'learning_rate': 0.010102762119397351, 'subsample': 0.44196584880757206, 'colsample_bytree': 0.4248188904455491, 'min_child_weight': 10, 'gamma': 0.5969285992367312}. Best is trial 32 with value: 0.6264966677245978.
[I 2025-01-19 17:32:47,999] Trial 45 finished with value: 0.6040205942757195 and parameters: {'n_estimators': 196, 'max_depth': 13, 'learning_rate': 0.03608175261846887, 'subsample': 0.49910767035199777, 'colsample_bytree': 0.41498423270934426, 'min_child_weight': 9, 'gamma': 0.5862393696733569}. Best is trial 32 with value: 0.6264966677245978.
[I 2025-01-19 17:32:53,903] Trial 46 finished with value: 0.5783772117349356 and parameters: {'n_estimators': 170, 'max_depth': 11, 'learning_rate': 0.23456652222557273, 'subsample': 0.4636905999694743, 'colsample_bytree': 0.4178713154341252, 'min_child_weight': 10, 'gamma': 0.546590902846219}. Best is trial 32 with value: 0.6264966677245978.
[I 2025-01-19 17:33:04,585] Trial 47 finished with value: 0.6042100830851654 and parameters: {'n_estimators': 178, 'max_depth': 15, 'learning_rate': 0.026469119129894258, 'subsample': 0.5532522820572875, 'colsample_bytree': 0.43899764912950584, 'min_child_weight': 3, 'gamma': 0.5135489137349264}. Best is trial 32 with value: 0.6264966677245978.
[I 2025-01-19 17:33:11,548] Trial 48 finished with value: 0.5937987905453926 and parameters: {'n_estimators': 207, 'max_depth': 10, 'learning_rate': 0.05661097160039319, 'subsample': 0.6190881855013768, 'colsample_bytree': 0.4010798914487499, 'min_child_weight': 10, 'gamma': 0.49021992631377664}. Best is trial 32 with value: 0.6264966677245978.
[I 2025-01-19 17:33:24,434] Trial 49 finished with value: 0.6108463285154062 and parameters: {'n_estimators': 282, 'max_depth': 14, 'learning_rate': 0.011170895054182253, 'subsample': 0.4001980627038158, 'colsample_bytree': 0.7312179862068469, 'min_child_weight': 6, 'gamma': 0.5993549586087356}. Best is trial 32 with value: 0.6264966677245978.
Best Parameters (Temporal Set): {'n_estimators': 189, 'max_depth': 13, 'learning_rate': 0.010526222378915213, 'subsample': 0.42496204179679387, 'colsample_bytree': 0.43441702668124493, 'min_child_weight': 10, 'gamma': 0.41175262428544657}
Best F1 Score (Temporal Set): 0.6264966677245978
c:\Users\msegh\MaissaSeghir_thesis\XGBoost Trees (no proxies).py:186: ExperimentalWarning: plot_optimization_history is experimental (supported from v2.2.0). The interface can change in the future.
  optuna.visualization.matplotlib.plot_optimization_history(study_temp)

Validation Metrics (Temporal Set):
Accuracy: 0.571258199165176
Precision: 0.4648620510150963
Recall: 0.6853415195702226
F1-Score: 0.5539702233250621
AUC-ROC: 0.5920613009845261
Confusion Matrix (Temporal Set):
[[1023 1028]
 [ 410  893]]

Test Metrics (Random Set - XGBoost):
Accuracy: 0.6124031007751938
Precision: 0.5632949727437916
Recall: 0.6163021868787276
F1-Score: 0.5886075949367088
AUC-ROC: 0.6646825460965013
Confusion Matrix:
[[1124  721]
 [ 579  930]]

Test Metrics (Temporal Set - XGBoost):
Accuracy: 0.6627906976744186
Precision: 0.6371571072319202
Recall: 0.37823834196891193
F1-Score: 0.4746864839758477
AUC-ROC: 0.7158730815693559
Confusion Matrix:
[[1712  291]
 [ 840  511]]

 """