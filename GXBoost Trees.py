# Import necessary libraries
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, KFold, TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the balanced temporal training and validation datasets
X_temp_balanced = pd.read_csv('X_train_temp.csv')
y_temp_balanced = pd.read_csv('y_train_temp.csv')
x_val_temp = pd.read_csv('x_val_temp.csv')
y_val_temp = pd.read_csv('y_val_temp.csv')

# Load the balanced random training and validation datasets
X_rand_balanced = pd.read_csv('X_train_rand.csv')
y_rand_balanced = pd.read_csv('y_train_rand.csv')
x_val_rand = pd.read_csv('x_val_rand.csv')
y_val_rand = pd.read_csv('y_val_rand.csv')

# Initialize the XGBoost classifier with randomstate 777
xgb_model = XGBClassifier(random_state=777, eval_metric='logloss') 

# Set up 5-fold cross-validation for the random set
kf = KFold(n_splits=5, shuffle=True, random_state=777)

# Perform cross-validation on the random set
random_cv_scores = cross_val_score(xgb_model, X_rand_balanced, y_rand_balanced, cv=kf, scoring='f1')
print("Random Set - 5-Fold Cross-Validation F1 Scores:", random_cv_scores)
print("Random Set - Mean F1 Score:", np.mean(random_cv_scores))

# Set up time series cross-validation for the temporal dataset
tscv = TimeSeriesSplit(n_splits=5)

# Perform cross-validation on the temporal set
temporal_cv_scores = cross_val_score(xgb_model, X_temp_balanced, y_temp_balanced, cv=tscv, scoring='f1')
print("Temporal Set - Time Series Cross-Validation F1 Scores:", temporal_cv_scores)
print("Temporal Set - Mean F1 Score:", np.mean(temporal_cv_scores))

# Hyperparameter distribution for RandomizedSearchCV
param_dist = {
    'n_estimators': [50, 100, 150, 200, 250],
    'max_depth': [3, 5, 7, 10, 12],
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
    'subsample': [0.5, 0.6, 0.7, 0.8, 1.0],
    'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 1.0],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2, 0.3, 0.4]
}

# RandomizedSearchCV for Random Set
print("\nFitting RandomizedSearchCV for Random Set...")
random_search_rand = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=50,
    scoring='f1',
    cv=5,
    verbose=1,
    n_jobs=-1,
    random_state=777
)
random_search_rand.fit(X_rand_balanced, y_rand_balanced)
print("Best Parameters (Random Set):", random_search_rand.best_params_)
print("Best F1 Score (Random Set):", random_search_rand.best_score_)

# RandomizedSearchCV for Temporal Set
print("\nFitting RandomizedSearchCV for Temporal Set...")
random_search_temp = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=50,
    scoring='f1',
    cv=5,
    verbose=1,
    n_jobs=-1,
    random_state=777
)
random_search_temp.fit(X_temp_balanced, y_temp_balanced)
print("Best Parameters (Temporal Set):", random_search_temp.best_params_)
print("Best F1 Score (Temporal Set):", random_search_temp.best_score_)

# Model Evaluation on Validation Set
# Random Set
print("\nValidation Metrics (Random Set):")
best_model_rand = random_search_rand.best_estimator_
val_rand_predictions = best_model_rand.predict(x_val_rand)
val_rand_accuracy = accuracy_score(y_val_rand, val_rand_predictions)
val_rand_precision = precision_score(y_val_rand, val_rand_predictions)
val_rand_recall = recall_score(y_val_rand, val_rand_predictions)
val_rand_f1 = f1_score(y_val_rand, val_rand_predictions)
val_rand_auc = roc_auc_score(y_val_rand, val_rand_predictions)

print(f"Accuracy: {val_rand_accuracy}")
print(f"Precision: {val_rand_precision}")
print(f"Recall: {val_rand_recall}")
print(f"F1-Score: {val_rand_f1}")
print(f"AUC-ROC: {val_rand_auc}")

# Confusion Matrix for Random Set
conf_matrix_rand = confusion_matrix(y_val_rand, val_rand_predictions)
print("Confusion Matrix (Random Set):")
print(conf_matrix_rand)

# Temporal Set
print("\nValidation Metrics (Temporal Set):")
best_model_temp = random_search_temp.best_estimator_
val_temp_predictions = best_model_temp.predict(x_val_temp)
val_temp_accuracy = accuracy_score(y_val_temp, val_temp_predictions)
val_temp_precision = precision_score(y_val_temp, val_temp_predictions)
val_temp_recall = recall_score(y_val_temp, val_temp_predictions)
val_temp_f1 = f1_score(y_val_temp, val_temp_predictions)
val_temp_auc = roc_auc_score(y_val_temp, val_temp_predictions)

print(f"Accuracy: {val_temp_accuracy}")
print(f"Precision: {val_temp_precision}")
print(f"Recall: {val_temp_recall}")
print(f"F1-Score: {val_temp_f1}")
print(f"AUC-ROC: {val_temp_auc}")

# Confusion Matrix for Temporal Set
conf_matrix_temp = confusion_matrix(y_val_temp, val_temp_predictions)
print("Confusion Matrix (Temporal Set):")
print(conf_matrix_temp)

"""Random Set - 5-Fold Cross-Validation F1 Scores: [0.81913303 0.82719547 0.82200087 0.81839878 0.81979257]
Random Set - Mean F1 Score: 0.8213041439894152
Temporal Set - Time Series Cross-Validation F1 Scores: [0.85541126 0.86601775 0.85850144 0.32064985 0.        ]
Temporal Set - Mean F1 Score: 0.5801160592964267

Fitting RandomizedSearchCV for Random Set...
Fitting 5 folds for each of 50 candidates, totalling 250 fits
Best Parameters (Random Set): {'subsample': 0.6, 'n_estimators': 150, 'min_child_weight': 5, 'max_depth': 7, 'learning_rate': 0.01, 'gamma': 0.4, 'colsample_bytree': 0.5}
Best F1 Score (Random Set): 0.7109982122532565

Fitting RandomizedSearchCV for Temporal Set...
Fitting 5 folds for each of 50 candidates, totalling 250 fits
Best Parameters (Temporal Set): {'subsample': 0.5, 'n_estimators': 250, 'min_child_weight': 1, 'max_depth': 12, 'learning_rate': 0.1, 'gamma': 0.1, 'colsample_bytree': 0.6}
Best F1 Score (Temporal Set): 0.8002906258713021

Validation Metrics (Random Set):
Accuracy: 0.7771739130434783
Precision: 0.8817236255572065
Recall: 0.7486752460257381
F1-Score: 0.8097707423580787
AUC-ROC: 0.7875517529736191
Confusion Matrix (Random Set):
[[1895  398]
 [ 996 2967]]

Validation Metrics (Temporal Set):
Accuracy: 0.6360294117647058
Precision: 0.5836388634280477
Recall: 0.8470236115729963
F1-Score: 0.6910866910866911
AUC-ROC: 0.6438873059403917
Confusion Matrix (Temporal Set):
[[1432 1817]
 [ 460 2547]]"""