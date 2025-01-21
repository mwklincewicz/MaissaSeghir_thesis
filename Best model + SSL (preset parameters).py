from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.semi_supervised import LabelPropagation
from sklearn.inspection import permutation_importance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the balanced temporal training and validation datasets
X_temp_balanced = pd.read_csv('X_train_temp.csv')
y_temp_balanced = pd.read_csv('y_train_temp.csv')
x_val_temp = pd.read_csv('x_val_temp.csv')
y_val_temp = pd.read_csv('y_val_temp.csv')
X_test_temp = pd.read_csv('X_test_temp.csv')
y_test_temp = pd.read_csv('y_test_temp.csv')

# Load the unlabeled dataset
unlabeled_data_encoded_df = pd.read_csv('unlabeled_data_encoded_df.csv')

# Preset parameters for XGBoost based on the best performing model
best_params_temp = {
    'n_estimators': 164,
    'max_depth': 10,
    'learning_rate': 0.012810188915189686,
    'subsample': 0.6384235632080573,
    'colsample_bytree': 0.5112889091784111,
    'min_child_weight': 8,
    'gamma': 0.23449750378310394,
}

# Combine training and validation datasets for Temporal Set
X_train_val_temp = pd.concat([X_temp_balanced, x_val_temp], axis=0)
y_train_val_temp = pd.concat([y_temp_balanced, y_val_temp], axis=0)

# Train the final XGBoost model with the preset parameters
final_model_temp_xgb = XGBClassifier(**best_params_temp, random_state=777, eval_metric='logloss')
final_model_temp_xgb.fit(X_train_val_temp, y_train_val_temp)

# Permutation Importance for the Temporal Set
perm_importance_temp_xgb = permutation_importance(final_model_temp_xgb, x_val_temp, y_val_temp, n_repeats=10, random_state=777)
importance_temp_df_xgb = pd.DataFrame({
    'Feature': X_temp_balanced.columns,
    'Importance': perm_importance_temp_xgb.importances_mean
})

# Plot Permutation Importance for the Temporal Set
importance_temp_df_xgb = importance_temp_df_xgb[(importance_temp_df_xgb['Importance'] > 0.001) | (importance_temp_df_xgb['Importance'] < -0.001)]
importance_temp_df_xgb.sort_values(by='Importance', ascending=False, inplace=True)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_temp_df_xgb)
plt.title('Permutation Importance (Temporal Set) - XGBoost + SSL')
plt.xlabel('Mean Decrease in Accuracy')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# Semi-Supervised Learning Experiment with SSL
lp_model = LabelPropagation(kernel='knn', n_neighbors=5)

# Combine labeled and unlabeled data
X_combined = pd.concat([X_train_val_temp, unlabeled_data_encoded_df], axis=0)
y_combined = pd.concat([y_train_val_temp['Target_binary'], pd.Series([-1] * len(unlabeled_data_encoded_df))], axis=0)  # -1 in this case represents unlabeled data

# Fit the Label Propagation model
lp_model.fit(X_combined, y_combined)

# Get the pseudo-labels for the unlabeled df
pseudo_labels = lp_model.transduction_[len(X_train_val_temp):]

# Combine the labeled data with pseudo-labeled data
X_train_pseudo = pd.concat([X_train_val_temp, unlabeled_data_encoded_df], axis=0)
y_train_pseudo = pd.concat([y_train_val_temp['Target_binary'], pd.Series(pseudo_labels)], axis=0)

# Train a new XGBoost model on the combined data with pseudo-labels
final_model_ssl_xgb = XGBClassifier(**best_params_temp, scale_pos_weight=1.5, random_state=777, eval_metric='logloss')
final_model_ssl_xgb.fit(X_train_pseudo, y_train_pseudo)

# Test Set Evaluation for the SSL model
test_ssl_predictions_xgb = final_model_ssl_xgb.predict(X_test_temp)
test_ssl_f1_xgb = f1_score(y_test_temp, test_ssl_predictions_xgb)
test_ssl_accuracy_xgb = accuracy_score(y_test_temp, test_ssl_predictions_xgb)
test_ssl_precision_xgb = precision_score(y_test_temp, test_ssl_predictions_xgb)
test_ssl_recall_xgb = recall_score(y_test_temp, test_ssl_predictions_xgb)
test_ssl_roc_auc_xgb = roc_auc_score(y_test_temp, final_model_ssl_xgb.predict_proba(X_test_temp)[:, 1])
test_ssl_conf_matrix_xgb = confusion_matrix(y_test_temp, test_ssl_predictions_xgb)

print("\nTest Metrics (SSL with XGBoost):")
print(f"Accuracy: {test_ssl_accuracy_xgb}")
print(f"Precision: {test_ssl_precision_xgb}")
print(f"Recall: {test_ssl_recall_xgb}")
print(f"F1-Score: {test_ssl_f1_xgb}")
print(f"AUC-ROC: {test_ssl_roc_auc_xgb}")
print(f"Confusion Matrix:\n{test_ssl_conf_matrix_xgb}")
