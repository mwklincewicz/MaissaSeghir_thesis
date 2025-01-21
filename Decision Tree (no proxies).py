# Importing libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.tree import plot_tree
import optuna #doing baysian optimalization instead of gridsearch 

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

# Load the balanced temporal datasets
X_temp_balanced = pd.read_csv('X_train_temp.csv')
y_temp_balanced = pd.read_csv('y_train_temp.csv')
x_val_temp = pd.read_csv('x_val_temp.csv')
y_val_temp = pd.read_csv('y_val_temp.csv')
x_test_temp = pd.read_csv('x_test_temp.csv')
y_test_temp = pd.read_csv('y_test_temp.csv')

# Load the balanced random datasets
X_rand_balanced = pd.read_csv('X_train_rand.csv')
y_rand_balanced = pd.read_csv('y_train_rand.csv')
x_val_rand = pd.read_csv('x_val_rand.csv')
y_val_rand = pd.read_csv('y_val_rand.csv')
x_test_rand = pd.read_csv('x_test_rand.csv')
y_test_rand = pd.read_csv('y_test_rand.csv')

# Remove proxy features from the temporal datasets
X_temp_balanced = X_temp_balanced.drop(columns=features_to_remove, errors='ignore')
x_val_temp = x_val_temp.drop(columns=features_to_remove, errors='ignore')
x_test_temp = x_test_temp.drop(columns=features_to_remove, errors='ignore')

# Remove proxy features from the random datasets
X_rand_balanced = X_rand_balanced.drop(columns=features_to_remove, errors='ignore')
x_val_rand = x_val_rand.drop(columns=features_to_remove, errors='ignore')
x_test_rand = x_test_rand.drop(columns=features_to_remove, errors='ignore')

# Custom function to create stratified time-series splits
def stratified_time_series_split(X, y, n_splits=5):
    # Create a list of indices to hold the splits
    indices = []

    # Initialize the StratifiedKFold
    stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=False)

    # Generate the indices for stratified time-series split
    for train_index, val_index in stratified_kfold.split(X, y):
        # Ensure the maintainance of the temporal order
        # slice the indices based on time (first train, then test)
        indices.append((train_index, val_index))
        
    return indices

# Initialize the decision tree classifier with random state 777
def objective(trial):
    # Define the hyperparameters to tune
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
    max_depth = trial.suggest_int('max_depth', 2, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)

    # Create the model with suggested hyperparameters
    model = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=777
    )

    # Perform cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=777)
    f1_scores = cross_val_score(model, X_rand_balanced, y_rand_balanced, cv=kf, scoring='f1')

    # Print the F1 scores for each fold
    print(f"F1 scores across folds (Random Set): {f1_scores}")
    print(f"Mean F1 score (Random Set): {np.mean(f1_scores)}")

    # Return the mean F1 score as the objective to minimize
    return np.mean(f1_scores)

# Optimize hyperparameters for the random set
study_rand = optuna.create_study(direction='maximize', study_name='Random Set Optimization')
study_rand.optimize(objective, n_trials=50, n_jobs=-1)

# Print the best parameters and score for the random set
print("Best Parameters (Random Set):", study_rand.best_params)
print("Best F1 Score (Random Set):", study_rand.best_value)

# Define a separate objective for the temporal set
def objective_temp(trial):
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
    max_depth = trial.suggest_int('max_depth', 2, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)

    model = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=777
    )

    stratified_splits = stratified_time_series_split(X_temp_balanced, y_temp_balanced, n_splits=5)
    f1_scores = []
    for train_index, val_index in stratified_splits:
        X_train, X_val = X_temp_balanced.iloc[train_index], X_temp_balanced.iloc[val_index]
        y_train, y_val = y_temp_balanced.iloc[train_index], y_temp_balanced.iloc[val_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        f1_scores.append(f1_score(y_val, y_pred))

    # Print the F1 scores for each fold
    print(f"F1 scores across folds (Temporal Set): {f1_scores}")
    print(f"Mean F1 score (Temporal Set): {np.mean(f1_scores)}")

    return np.mean(f1_scores)

# Optimize hyperparameters for the temporal set
study_temp = optuna.create_study(direction='maximize', study_name='Temporal Set Optimization')
study_temp.optimize(objective_temp, n_trials=50, n_jobs=-1)

# Print the best parameters and score for the temporal set
print("Best Parameters (Temporal Set):", study_temp.best_params)
print("Best F1 Score (Temporal Set):", study_temp.best_value)

# Random Validation Set 
best_model_rand = DecisionTreeClassifier(**study_rand.best_params, random_state=777)
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
best_model_temp = DecisionTreeClassifier(**study_temp.best_params, random_state=777)
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

# Combine the training and validation datasets for both random and temporal splits
X_combined_rand = pd.concat([X_rand_balanced, x_val_rand], axis=0)
y_combined_rand = pd.concat([y_rand_balanced, y_val_rand], axis=0)

X_combined_temp = pd.concat([X_temp_balanced, x_val_temp], axis=0)
y_combined_temp = pd.concat([y_temp_balanced, y_val_temp], axis=0)

# final model fit on the best models on the combined training and validation sets
final_model_rand = DecisionTreeClassifier(**study_rand.best_params, random_state=777)
final_model_rand.fit(X_combined_rand, y_combined_rand)

final_model_temp = DecisionTreeClassifier(**study_temp.best_params, random_state=777)
final_model_temp.fit(X_combined_temp, y_combined_temp)

# Permutation Importance for Random Set
perm_importance_rand = permutation_importance(final_model_rand, x_val_rand, y_val_rand, n_repeats=10, random_state=777)
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
perm_importance_temp = permutation_importance(final_model_temp, x_val_temp, y_val_temp, n_repeats=10, random_state=777)
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

# Visualization of Optimization History
plt.figure(figsize=(10, 6))
optuna.visualization.matplotlib.plot_optimization_history(study_rand)
plt.title('Optimization History DT (Random Set)')
plt.show()

plt.figure(figsize=(10, 6))
optuna.visualization.matplotlib.plot_optimization_history(study_temp)
plt.title('Optimization History DT (Temporal Set)')
plt.show()

# FINAL EVALUATION ON THE TEST SET

# Test the best model from the Random Split on the Random Test Set
test_rand_predictions = final_model_rand.predict(x_test_rand)

# Calculate the evaluation metrics from the random split on the Random Test Set
accuracy_test_rand = accuracy_score(y_test_rand, test_rand_predictions)
precision_test_rand = precision_score(y_test_rand, test_rand_predictions)
recall_test_rand = recall_score(y_test_rand, test_rand_predictions)
f1_test_rand = f1_score(y_test_rand, test_rand_predictions)
roc_auc_test_rand = roc_auc_score(y_test_rand, final_model_rand.predict_proba(x_test_rand)[:, 1])
conf_matrix_test_rand = confusion_matrix(y_test_rand, test_rand_predictions)

# Print the test metrics for the Random Test Set
print("Test Metrics (Random Set):")
print(f"Accuracy: {accuracy_test_rand}")
print(f"Precision: {precision_test_rand}")
print(f"Recall: {recall_test_rand}")
print(f"F1-Score: {f1_test_rand}")
print(f"AUC-ROC: {roc_auc_test_rand}")
print(f"Confusion Matrix:\n{conf_matrix_test_rand}")

# Test the best model from the Temporal Split on the Temporal Test Set
test_temp_predictions = final_model_temp.predict(x_test_temp)

# Calculate the evaluation metrics from the temporal split on the Temporal Test Set
accuracy_test_temp = accuracy_score(y_test_temp, test_temp_predictions)
precision_test_temp = precision_score(y_test_temp, test_temp_predictions)
recall_test_temp = recall_score(y_test_temp, test_temp_predictions)
f1_test_temp = f1_score(y_test_temp, test_temp_predictions)
roc_auc_test_temp = roc_auc_score(y_test_temp, final_model_temp.predict_proba(x_test_temp)[:, 1])
conf_matrix_test_temp = confusion_matrix(y_test_temp, test_temp_predictions)

# Print the test metrics for the Temporal Test Set
print("Test Metrics (Temporal Set):")
print(f"Accuracy: {accuracy_test_temp}")
print(f"Precision: {precision_test_temp}")
print(f"Recall: {recall_test_temp}")
print(f"F1-Score: {f1_test_temp}")
print(f"AUC-ROC: {roc_auc_test_temp}")
print(f"Confusion Matrix:\n{conf_matrix_test_temp}")

"""RESULTS WITHOUT THE PROXIES
F1 scores across folds (Random Set): [0.39752407 0.5942623  0.57962413 0.59829915 0.58954501]
Mean F1 score (Random Set): 0.5518509330449335
[I 2025-01-19 16:07:16,472] A new study created in memory with name: Random Set Optimization
[I 2025-01-19 16:07:17,403] Trial 2 finished with value: 0.6082006580228909 and parameters: {'criterion': 'entropy', 'max_depth': 4, 'min_samples_split': 11, 'min_samples_leaf': 19}. Best is trial 2 with value: 0.6082006580228909.
[I 2025-01-19 16:07:17,936] Trial 3 finished with value: 0.5788421522617615 and parameters: {'criterion': 'entropy', 'max_depth': 7, 'min_samples_split': 7, 'min_samples_leaf': 3}. Best is trial 2 with value: 0.6082006580228909.
[I 2025-01-19 16:07:18,038] Trial 1 finished with value: 0.5701971385552174 and parameters: {'criterion': 'gini', 'max_depth': 8, 'min_samples_split': 9, 'min_samples_leaf': 14}. Best is trial 2 with value: 0.6082006580228909.
[I 2025-01-19 16:07:18,916] Trial 0 finished with value: 0.5282010826549033 and parameters: {'criterion': 'gini', 'max_depth': 20, 'min_samples_split': 20, 'min_samples_leaf': 9}. Best is trial 2 with value: 0.6082006580228909.
[I 2025-01-19 16:07:19,392] Trial 4 finished with value: 0.5446313473502062 and parameters: {'criterion': 'gini', 'max_depth': 17, 'min_samples_split': 16, 'min_samples_leaf': 18}. Best is trial 2 with value: 0.6082006580228909.
[I 2025-01-19 16:07:20,031] Trial 6 finished with value: 0.5437511065382472 and parameters: {'criterion': 'gini', 'max_depth': 19, 'min_samples_split': 12, 'min_samples_leaf': 18}. Best is trial 2 with value: 0.6082006580228909.
[I 2025-01-19 16:07:20,318] Trial 5 finished with value: 0.552191724385873 and parameters: {'criterion': 'entropy', 'max_depth': 16, 'min_samples_split': 18, 'min_samples_leaf': 15}. Best is trial 2 with value: 0.6082006580228909.
[I 2025-01-19 16:07:21,074] Trial 7 finished with value: 0.5400976787163616 and parameters: {'criterion': 'gini', 'max_depth': 20, 'min_samples_split': 11, 'min_samples_leaf': 13}. Best is trial 2 with value: 0.6082006580228909.
[I 2025-01-19 16:07:21,686] Trial 10 finished with value: 0.5904187345540362 and parameters: {'criterion': 'entropy', 'max_depth': 6, 'min_samples_split': 9, 'min_samples_leaf': 15}. Best is trial 2 with value: 0.6082006580228909.
[I 2025-01-19 16:07:22,365] Trial 8 finished with value: 0.5392373861439761 and parameters: {'criterion': 'entropy', 'max_depth': 17, 'min_samples_split': 7, 'min_samples_leaf': 3}. Best is trial 2 with value: 0.6082006580228909.
[I 2025-01-19 16:07:22,584] Trial 9 finished with value: 0.5310502679211018 and parameters: {'criterion': 'gini', 'max_depth': 19, 'min_samples_split': 17, 'min_samples_leaf': 5}. Best is trial 2 with value: 0.6082006580228909.
[I 2025-01-19 16:07:23,007] Trial 13 finished with value: 0.5518509330449335 and parameters: {'criterion': 'entropy', 'max_depth': 2, 'min_samples_split': 3, 'min_samples_leaf': 8}. Best is trial 2 with value: 0.6082006580228909.
[I 2025-01-19 16:07:23,214] Trial 14 finished with value: 0.5518509330449335 and parameters: {'criterion': 'entropy', 'max_depth': 2, 'min_samples_split': 3, 'min_samples_leaf': 20}. Best is trial 2 with value: 0.6082006580228909.
[I 2025-01-19 16:07:23,598] Trial 11 finished with value: 0.5479849519839318 and parameters: {'criterion': 'entropy', 'max_depth': 13, 'min_samples_split': 12, 'min_samples_leaf': 3}. Best is trial 2 with value: 0.6082006580228909.
[I 2025-01-19 16:07:23,775] Trial 15 finished with value: 0.6255265923352828 and parameters: {'criterion': 'entropy', 'max_depth': 3, 'min_samples_split': 13, 'min_samples_leaf': 19}. Best is trial 15 with value: 0.6255265923352828.
[I 2025-01-19 16:07:24,007] Trial 12 finished with value: 0.5533925647592028 and parameters: {'criterion': 'gini', 'max_depth': 14, 'min_samples_split': 8, 'min_samples_leaf': 5}. Best is trial 15 with value: 0.6255265923352828.
[I 2025-01-19 16:07:24,421] Trial 16 finished with value: 0.5877616999699969 and parameters: {'criterion': 'entropy', 'max_depth': 6, 'min_samples_split': 13, 'min_samples_leaf': 16}. Best is trial 15 with value: 0.6255265923352828.
[I 2025-01-19 16:07:24,817] Trial 17 finished with value: 0.5877616999699969 and parameters: {'criterion': 'entropy', 'max_depth': 6, 'min_samples_split': 14, 'min_samples_leaf': 16}. Best is trial 15 with value: 0.6255265923352828.
[I 2025-01-19 16:07:24,962] Trial 18 finished with value: 0.5937564109921429 and parameters: {'criterion': 'entropy', 'max_depth': 5, 'min_samples_split': 14, 'min_samples_leaf': 20}. Best is trial 15 with value: 0.6255265923352828.
[I 2025-01-19 16:07:25,111] Trial 19 finished with value: 0.5937564109921429 and parameters: {'criterion': 'entropy', 'max_depth': 5, 'min_samples_split': 14, 'min_samples_leaf': 20}. Best is trial 15 with value: 0.6255265923352828.
[I 2025-01-19 16:07:25,423] Trial 20 finished with value: 0.6074244053731709 and parameters: {'criterion': 'entropy', 'max_depth': 4, 'min_samples_split': 15, 'min_samples_leaf': 20}. Best is trial 15 with value: 0.6255265923352828.
[I 2025-01-19 16:07:26,202] Trial 24 finished with value: 0.6255265923352828 and parameters: {'criterion': 'entropy', 'max_depth': 3, 'min_samples_split': 16, 'min_samples_leaf': 12}. Best is trial 15 with value: 0.6255265923352828.
[I 2025-01-19 16:07:26,658] Trial 21 finished with value: 0.5616505903249923 and parameters: {'criterion': 'entropy', 'max_depth': 10, 'min_samples_split': 15, 'min_samples_leaf': 20}. Best is trial 15 with value: 0.6255265923352828.
[I 2025-01-19 16:07:26,802] Trial 22 finished with value: 0.5623949435070499 and parameters: {'criterion': 'entropy', 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 12}. Best is trial 15 with value: 0.6255265923352828.
[I 2025-01-19 16:07:26,852] Trial 23 finished with value: 0.5709870996894835 and parameters: {'criterion': 'entropy', 'max_depth': 9, 'min_samples_split': 5, 'min_samples_leaf': 11}. Best is trial 15 with value: 0.6255265923352828.
[I 2025-01-19 16:07:27,662] Trial 26 finished with value: 0.6255265923352828 and parameters: {'criterion': 'entropy', 'max_depth': 3, 'min_samples_split': 19, 'min_samples_leaf': 11}. Best is trial 15 with value: 0.6255265923352828.
[I 2025-01-19 16:07:27,735] Trial 27 finished with value: 0.6255265923352828 and parameters: {'criterion': 'entropy', 'max_depth': 3, 'min_samples_split': 19, 'min_samples_leaf': 18}. Best is trial 15 with value: 0.6255265923352828.
[I 2025-01-19 16:07:27,838] Trial 28 finished with value: 0.6255265923352828 and parameters: {'criterion': 'entropy', 'max_depth': 3, 'min_samples_split': 18, 'min_samples_leaf': 18}. Best is trial 15 with value: 0.6255265923352828.
[I 2025-01-19 16:07:28,094] Trial 25 finished with value: 0.5754085681649854 and parameters: {'criterion': 'entropy', 'max_depth': 9, 'min_samples_split': 18, 'min_samples_leaf': 12}. Best is trial 15 with value: 0.6255265923352828.
[I 2025-01-19 16:07:28,402] Trial 30 finished with value: 0.5518509330449335 and parameters: {'criterion': 'entropy', 'max_depth': 2, 'min_samples_split': 17, 'min_samples_leaf': 10}. Best is trial 15 with value: 0.6255265923352828.
[I 2025-01-19 16:07:28,495] Trial 29 finished with value: 0.6255265923352828 and parameters: {'criterion': 'entropy', 'max_depth': 3, 'min_samples_split': 20, 'min_samples_leaf': 9}. Best is trial 15 with value: 0.6255265923352828.
[I 2025-01-19 16:07:29,526] Trial 34 finished with value: 0.6177447878947895 and parameters: {'criterion': 'entropy', 'max_depth': 4, 'min_samples_split': 20, 'min_samples_leaf': 7}. Best is trial 15 with value: 0.6255265923352828.
[I 2025-01-19 16:07:30,110] Trial 31 finished with value: 0.5595198384785487 and parameters: {'criterion': 'entropy', 'max_depth': 12, 'min_samples_split': 20, 'min_samples_leaf': 9}. Best is trial 15 with value: 0.6255265923352828.
[I 2025-01-19 16:07:30,421] Trial 35 finished with value: 0.6255265923352828 and parameters: {'criterion': 'entropy', 'max_depth': 3, 'min_samples_split': 19, 'min_samples_leaf': 17}. Best is trial 15 with value: 0.6255265923352828.
[I 2025-01-19 16:07:30,494] Trial 32 finished with value: 0.5618003256372439 and parameters: {'criterion': 'entropy', 'max_depth': 12, 'min_samples_split': 19, 'min_samples_leaf': 9}. Best is trial 15 with value: 0.6255265923352828.
[I 2025-01-19 16:07:30,853] Trial 33 finished with value: 0.5536089806006503 and parameters: {'criterion': 'entropy', 'max_depth': 12, 'min_samples_split': 20, 'min_samples_leaf': 8}. Best is trial 15 with value: 0.6255265923352828.
[I 2025-01-19 16:07:31,037] Trial 36 finished with value: 0.6255265923352828 and parameters: {'criterion': 'entropy', 'max_depth': 3, 'min_samples_split': 19, 'min_samples_leaf': 17}. Best is trial 15 with value: 0.6255265923352828.
[I 2025-01-19 16:07:31,801] Trial 37 finished with value: 0.5613054763672471 and parameters: {'criterion': 'entropy', 'max_depth': 7, 'min_samples_split': 16, 'min_samples_leaf': 14}. Best is trial 15 with value: 0.6255265923352828.
[I 2025-01-19 16:07:31,961] Trial 38 finished with value: 0.5701971385552174 and parameters: {'criterion': 'gini', 'max_depth': 8, 'min_samples_split': 16, 'min_samples_leaf': 14}. Best is trial 15 with value: 0.6255265923352828.
[I 2025-01-19 16:07:32,171] Trial 39 finished with value: 0.5763784328623827 and parameters: {'criterion': 'gini', 'max_depth': 7, 'min_samples_split': 16, 'min_samples_leaf': 13}. Best is trial 15 with value: 0.6255265923352828.
[I 2025-01-19 16:07:32,369] Trial 40 finished with value: 0.5707607101593665 and parameters: {'criterion': 'gini', 'max_depth': 7, 'min_samples_split': 16, 'min_samples_leaf': 14}. Best is trial 15 with value: 0.6255265923352828.
[I 2025-01-19 16:07:32,894] Trial 41 finished with value: 0.6271853211350727 and parameters: {'criterion': 'gini', 'max_depth': 5, 'min_samples_split': 16, 'min_samples_leaf': 14}. Best is trial 41 with value: 0.6271853211350727.
[I 2025-01-19 16:07:33,139] Trial 42 finished with value: 0.6275947676517447 and parameters: {'criterion': 'gini', 'max_depth': 5, 'min_samples_split': 17, 'min_samples_leaf': 11}. Best is trial 42 with value: 0.6275947676517447.
[I 2025-01-19 16:07:33,242] Trial 43 finished with value: 0.6062340485997804 and parameters: {'criterion': 'gini', 'max_depth': 4, 'min_samples_split': 10, 'min_samples_leaf': 11}. Best is trial 42 with value: 0.6275947676517447.
[I 2025-01-19 16:07:33,497] Trial 44 finished with value: 0.6082006580228909 and parameters: {'criterion': 'entropy', 'max_depth': 4, 'min_samples_split': 18, 'min_samples_leaf': 18}. Best is trial 42 with value: 0.6275947676517447.
[I 2025-01-19 16:07:34,238] Trial 45 finished with value: 0.6275947676517447 and parameters: {'criterion': 'gini', 'max_depth': 5, 'min_samples_split': 17, 'min_samples_leaf': 11}. Best is trial 42 with value: 0.6275947676517447.
[I 2025-01-19 16:07:34,330] Trial 46 finished with value: 0.6275947676517447 and parameters: {'criterion': 'gini', 'max_depth': 5, 'min_samples_split': 17, 'min_samples_leaf': 11}. Best is trial 42 with value: 0.6275947676517447.
[I 2025-01-19 16:07:34,477] Trial 47 finished with value: 0.6272954073886691 and parameters: {'criterion': 'gini', 'max_depth': 5, 'min_samples_split': 17, 'min_samples_leaf': 12}. Best is trial 42 with value: 0.6275947676517447.
[I 2025-01-19 16:07:34,674] Trial 48 finished with value: 0.6272954073886691 and parameters: {'criterion': 'gini', 'max_depth': 5, 'min_samples_split': 12, 'min_samples_leaf': 12}. Best is trial 42 with value: 0.6275947676517447.
[I 2025-01-19 16:07:35,077] Trial 49 finished with value: 0.6272954073886691 and parameters: {'criterion': 'gini', 'max_depth': 5, 'min_samples_split': 12, 'min_samples_leaf': 12}. Best is trial 42 with value: 0.6275947676517447.
Best Parameters (Random Set): {'criterion': 'gini', 'max_depth': 5, 'min_samples_split': 17, 'min_samples_leaf': 11}
Best F1 Score (Random Set): 0.6275947676517447

F1 scores across folds (Temporal Set): [0.4911032028469751, 0.5453629032258064, 0.6533383628819313, 0.5820153637596024, 0.6093220338983052]
Mean F1 score (Temporal Set): 0.5762283733225241
[I 2025-01-19 16:07:35,090] A new study created in memory with name: Temporal Set Optimization
[I 2025-01-19 16:07:36,042] Trial 0 finished with value: 0.6346833760744262 and parameters: {'criterion': 'gini', 'max_depth': 5, 'min_samples_split': 11, 'min_samples_leaf': 13}. Best is trial 0 with value: 0.6346833760744262.
[I 2025-01-19 16:07:37,006] Trial 1 finished with value: 0.5740211667786668 and parameters: {'criterion': 'gini', 'max_depth': 13, 'min_samples_split': 12, 'min_samples_leaf': 9}. Best is trial 0 with value: 0.6346833760744262.
[I 2025-01-19 16:07:37,191] Trial 2 finished with value: 0.5667281005374577 and parameters: {'criterion': 'entropy', 'max_depth': 13, 'min_samples_split': 4, 'min_samples_leaf': 12}. Best is trial 0 with value: 0.6346833760744262.
[I 2025-01-19 16:07:37,260] Trial 3 finished with value: 0.5670674738487056 and parameters: {'criterion': 'gini', 'max_depth': 14, 'min_samples_split': 15, 'min_samples_leaf': 3}. Best is trial 0 with value: 0.6346833760744262.
[I 2025-01-19 16:07:37,799] Trial 7 finished with value: 0.6517566854242691 and parameters: {'criterion': 'gini', 'max_depth': 2, 'min_samples_split': 18, 'min_samples_leaf': 20}. Best is trial 7 with value: 0.6517566854242691.
[I 2025-01-19 16:07:38,004] Trial 4 finished with value: 0.5479503097645682 and parameters: {'criterion': 'gini', 'max_depth': 15, 'min_samples_split': 19, 'min_samples_leaf': 15}. Best is trial 7 with value: 0.6517566854242691.
[I 2025-01-19 16:07:38,987] Trial 6 finished with value: 0.5984522254688168 and parameters: {'criterion': 'entropy', 'max_depth': 10, 'min_samples_split': 13, 'min_samples_leaf': 10}. Best is trial 7 with value: 0.6517566854242691.
[I 2025-01-19 16:07:39,040] Trial 9 finished with value: 0.6244375851758084 and parameters: {'criterion': 'gini', 'max_depth': 6, 'min_samples_split': 12, 'min_samples_leaf': 18}. Best is trial 7 with value: 0.6517566854242691.
[I 2025-01-19 16:07:39,084] Trial 5 finished with value: 0.5625102373396673 and parameters: {'criterion': 'gini', 'max_depth': 13, 'min_samples_split': 11, 'min_samples_leaf': 1}. Best is trial 7 with value: 0.6517566854242691.
[I 2025-01-19 16:07:39,848] Trial 11 finished with value: 0.6396154694987368 and parameters: {'criterion': 'gini', 'max_depth': 4, 'min_samples_split': 8, 'min_samples_leaf': 4}. Best is trial 7 with value: 0.6517566854242691.
[I 2025-01-19 16:07:40,045] Trial 8 finished with value: 0.5662268338868184 and parameters: {'criterion': 'entropy', 'max_depth': 14, 'min_samples_split': 17, 'min_samples_leaf': 4}. Best is trial 7 with value: 0.6517566854242691.
[I 2025-01-19 16:07:40,230] Trial 12 finished with value: 0.625168799450655 and parameters: {'criterion': 'entropy', 'max_depth': 6, 'min_samples_split': 8, 'min_samples_leaf': 10}. Best is trial 7 with value: 0.6517566854242691.
[I 2025-01-19 16:07:40,646] Trial 14 finished with value: 0.6517566854242691 and parameters: {'criterion': 'gini', 'max_depth': 2, 'min_samples_split': 6, 'min_samples_leaf': 20}. Best is trial 7 with value: 0.6517566854242691.
[I 2025-01-19 16:07:40,825] Trial 10 finished with value: 0.5935511518600685 and parameters: {'criterion': 'entropy', 'max_depth': 10, 'min_samples_split': 2, 'min_samples_leaf': 4}. Best is trial 7 with value: 0.6517566854242691.
[I 2025-01-19 16:07:40,864] Trial 15 finished with value: 0.6517566854242691 and parameters: {'criterion': 'gini', 'max_depth': 2, 'min_samples_split': 7, 'min_samples_leaf': 20}. Best is trial 7 with value: 0.6517566854242691.
[I 2025-01-19 16:07:41,315] Trial 16 finished with value: 0.6517566854242691 and parameters: {'criterion': 'gini', 'max_depth': 2, 'min_samples_split': 4, 'min_samples_leaf': 20}. Best is trial 7 with value: 0.6517566854242691.
[I 2025-01-19 16:07:41,489] Trial 17 finished with value: 0.6517566854242691 and parameters: {'criterion': 'gini', 'max_depth': 2, 'min_samples_split': 20, 'min_samples_leaf': 20}. Best is trial 7 with value: 0.6517566854242691.
[I 2025-01-19 16:07:42,045] Trial 13 finished with value: 0.5416226701052121 and parameters: {'criterion': 'entropy', 'max_depth': 19, 'min_samples_split': 20, 'min_samples_leaf': 20}. Best is trial 7 with value: 0.6517566854242691.
[I 2025-01-19 16:07:42,905] Trial 18 finished with value: 0.5386054474886744 and parameters: {'criterion': 'gini', 'max_depth': 20, 'min_samples_split': 20, 'min_samples_leaf': 17}. Best is trial 7 with value: 0.6517566854242691.
[I 2025-01-19 16:07:43,271] Trial 21 finished with value: 0.6161100788238159 and parameters: {'criterion': 'gini', 'max_depth': 7, 'min_samples_split': 16, 'min_samples_leaf': 16}. Best is trial 7 with value: 0.6517566854242691.
[I 2025-01-19 16:07:43,355] Trial 19 finished with value: 0.5389681520638024 and parameters: {'criterion': 'gini', 'max_depth': 19, 'min_samples_split': 18, 'min_samples_leaf': 16}. Best is trial 7 with value: 0.6517566854242691.
[I 2025-01-19 16:07:43,418] Trial 20 finished with value: 0.5389681520638024 and parameters: {'criterion': 'gini', 'max_depth': 19, 'min_samples_split': 16, 'min_samples_leaf': 16}. Best is trial 7 with value: 0.6517566854242691.
[I 2025-01-19 16:07:44,126] Trial 24 finished with value: 0.6371820074407094 and parameters: {'criterion': 'gini', 'max_depth': 3, 'min_samples_split': 8, 'min_samples_leaf': 18}. Best is trial 7 with value: 0.6517566854242691.
[I 2025-01-19 16:07:44,141] Trial 25 finished with value: 0.6371820074407094 and parameters: {'criterion': 'gini', 'max_depth': 3, 'min_samples_split': 8, 'min_samples_leaf': 18}. Best is trial 7 with value: 0.6517566854242691.
[I 2025-01-19 16:07:44,305] Trial 22 finished with value: 0.5936653995873818 and parameters: {'criterion': 'gini', 'max_depth': 8, 'min_samples_split': 15, 'min_samples_leaf': 16}. Best is trial 7 with value: 0.6517566854242691.
[I 2025-01-19 16:07:44,724] Trial 23 finished with value: 0.6088900105540019 and parameters: {'criterion': 'gini', 'max_depth': 8, 'min_samples_split': 6, 'min_samples_leaf': 7}. Best is trial 7 with value: 0.6517566854242691.
[I 2025-01-19 16:07:45,151] Trial 28 finished with value: 0.6353026681489486 and parameters: {'criterion': 'gini', 'max_depth': 4, 'min_samples_split': 6, 'min_samples_leaf': 14}. Best is trial 7 with value: 0.6517566854242691.
[I 2025-01-19 16:07:45,619] Trial 29 finished with value: 0.6353026681489486 and parameters: {'criterion': 'gini', 'max_depth': 4, 'min_samples_split': 6, 'min_samples_leaf': 13}. Best is trial 7 with value: 0.6517566854242691.
[I 2025-01-19 16:07:45,628] Trial 26 finished with value: 0.5893662463691542 and parameters: {'criterion': 'gini', 'max_depth': 8, 'min_samples_split': 6, 'min_samples_leaf': 19}. Best is trial 7 with value: 0.6517566854242691.
[I 2025-01-19 16:07:45,679] Trial 27 finished with value: 0.6088900105540019 and parameters: {'criterion': 'gini', 'max_depth': 8, 'min_samples_split': 6, 'min_samples_leaf': 7}. Best is trial 7 with value: 0.6517566854242691.
[I 2025-01-19 16:07:45,795] Trial 30 finished with value: 0.6517566854242691 and parameters: {'criterion': 'gini', 'max_depth': 2, 'min_samples_split': 6, 'min_samples_leaf': 19}. Best is trial 7 with value: 0.6517566854242691.
[I 2025-01-19 16:07:46,401] Trial 31 finished with value: 0.6517566854242691 and parameters: {'criterion': 'gini', 'max_depth': 2, 'min_samples_split': 2, 'min_samples_leaf': 19}. Best is trial 7 with value: 0.6517566854242691.
[I 2025-01-19 16:07:46,521] Trial 34 finished with value: 0.6517566854242691 and parameters: {'criterion': 'gini', 'max_depth': 2, 'min_samples_split': 3, 'min_samples_leaf': 20}. Best is trial 7 with value: 0.6517566854242691.
[I 2025-01-19 16:07:46,714] Trial 32 finished with value: 0.6347544458645589 and parameters: {'criterion': 'gini', 'max_depth': 5, 'min_samples_split': 9, 'min_samples_leaf': 19}. Best is trial 7 with value: 0.6517566854242691.
[I 2025-01-19 16:07:46,829] Trial 33 finished with value: 0.6347544458645589 and parameters: {'criterion': 'gini', 'max_depth': 5, 'min_samples_split': 9, 'min_samples_leaf': 19}. Best is trial 7 with value: 0.6517566854242691.
[I 2025-01-19 16:07:47,440] Trial 35 finished with value: 0.6345847812532546 and parameters: {'criterion': 'gini', 'max_depth': 5, 'min_samples_split': 4, 'min_samples_leaf': 20}. Best is trial 7 with value: 0.6517566854242691.
[I 2025-01-19 16:07:47,598] Trial 38 finished with value: 0.6371820074407094 and parameters: {'criterion': 'gini', 'max_depth': 3, 'min_samples_split': 4, 'min_samples_leaf': 17}. Best is trial 7 with value: 0.6517566854242691.
[I 2025-01-19 16:07:47,670] Trial 36 finished with value: 0.6345847812532546 and parameters: {'criterion': 'gini', 'max_depth': 5, 'min_samples_split': 10, 'min_samples_leaf': 20}. Best is trial 7 with value: 0.6517566854242691.
[I 2025-01-19 16:07:47,820] Trial 37 finished with value: 0.6342913295158544 and parameters: {'criterion': 'gini', 'max_depth': 5, 'min_samples_split': 4, 'min_samples_leaf': 17}. Best is trial 7 with value: 0.6517566854242691.
[I 2025-01-19 16:07:48,267] Trial 39 finished with value: 0.6371820074407094 and parameters: {'criterion': 'gini', 'max_depth': 3, 'min_samples_split': 4, 'min_samples_leaf': 17}. Best is trial 7 with value: 0.6517566854242691.
[I 2025-01-19 16:07:48,472] Trial 40 finished with value: 0.6400128897767673 and parameters: {'criterion': 'entropy', 'max_depth': 3, 'min_samples_split': 4, 'min_samples_leaf': 13}. Best is trial 7 with value: 0.6517566854242691.
[I 2025-01-19 16:07:48,572] Trial 41 finished with value: 0.6400128897767673 and parameters: {'criterion': 'entropy', 'max_depth': 3, 'min_samples_split': 4, 'min_samples_leaf': 14}. Best is trial 7 with value: 0.6517566854242691.
[I 2025-01-19 16:07:48,726] Trial 42 finished with value: 0.640348907899093 and parameters: {'criterion': 'entropy', 'max_depth': 3, 'min_samples_split': 5, 'min_samples_leaf': 12}. Best is trial 7 with value: 0.6517566854242691.
[I 2025-01-19 16:07:49,171] Trial 44 finished with value: 0.6517566854242691 and parameters: {'criterion': 'gini', 'max_depth': 2, 'min_samples_split': 19, 'min_samples_leaf': 20}. Best is trial 7 with value: 0.6517566854242691.
[I 2025-01-19 16:07:49,232] Trial 45 finished with value: 0.6517566854242691 and parameters: {'criterion': 'gini', 'max_depth': 2, 'min_samples_split': 13, 'min_samples_leaf': 18}. Best is trial 7 with value: 0.6517566854242691.
[I 2025-01-19 16:07:49,400] Trial 46 finished with value: 0.6517566854242691 and parameters: {'criterion': 'gini', 'max_depth': 2, 'min_samples_split': 19, 'min_samples_leaf': 18}. Best is trial 7 with value: 0.6517566854242691.
[I 2025-01-19 16:07:50,075] Trial 48 finished with value: 0.6362074126059474 and parameters: {'criterion': 'gini', 'max_depth': 4, 'min_samples_split': 18, 'min_samples_leaf': 20}. Best is trial 7 with value: 0.6517566854242691.
[I 2025-01-19 16:07:50,227] Trial 43 finished with value: 0.5679738203853333 and parameters: {'criterion': 'entropy', 'max_depth': 12, 'min_samples_split': 5, 'min_samples_leaf': 12}. Best is trial 7 with value: 0.6517566854242691.
[I 2025-01-19 16:07:50,836] Trial 47 finished with value: 0.5452708614425855 and parameters: {'criterion': 'gini', 'max_depth': 16, 'min_samples_split': 13, 'min_samples_leaf': 18}. Best is trial 7 with value: 0.6517566854242691.
[I 2025-01-19 16:07:50,994] Trial 49 finished with value: 0.5519888105898922 and parameters: {'criterion': 'gini', 'max_depth': 17, 'min_samples_split': 7, 'min_samples_leaf': 20}. Best is trial 7 with value: 0.6517566854242691.
Best Parameters (Temporal Set): {'criterion': 'gini', 'max_depth': 2, 'min_samples_split': 18, 'min_samples_leaf': 20}
Best F1 Score (Temporal Set): 0.6517566854242691
Validation Metrics (Random Set):
Accuracy: 0.5948121645796064
Precision: 0.56737998843262
Recall: 0.6162060301507538
F1-Score: 0.5907859078590786
AUC-ROC: 0.6375243128240522
Confusion Matrix:
[[1014  748]
 [ 611  981]]
Validation Metrics (Temporal Set):
Accuracy: 0.554561717352415
Precision: 0.45370819195346584
Recall: 0.718342287029931
F1-Score: 0.5561497326203209
AUC-ROC: 0.5900303952960071
Confusion Matrix:
[[ 924 1127]
 [ 367  936]]

Test Metrics (Random Set):
Accuracy: 0.5927251043530113
Precision: 0.5341288782816229
Recall: 0.7415506958250497
F1-Score: 0.6209766925638179
AUC-ROC: 0.6491772041643544
Confusion Matrix:
[[ 869  976]
 [ 390 1119]]
Test Metrics (Temporal Set):
Accuracy: 0.6547406082289803
Precision: 0.6241956241956242
Recall: 0.35899333826794966
F1-Score: 0.45582706766917297
AUC-ROC: 0.6572818418560169
Confusion Matrix:
[[1711  292]
 [ 866  485]]"""