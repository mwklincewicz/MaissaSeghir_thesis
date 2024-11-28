"""THIS IS THE LAST NOTEBOOK/EXPERIMENT
Using the best model in my SSL experiment to see if i can improve metrics, i am expecially interested in recall improvement. 

The best model was the XGBoost on a temporal split, which gave the following results on the test set:
Test Metrics (Temporal Set - XGBoost):
Accuracy: 0.7152653548002386
Precision: 0.6892925430210325
Recall: 0.533678756476684
F1-Score: 0.6015853149770546
AUC-ROC: 0.7802864541086224
Confusion Matrix:
[[1678  325]
 [ 630  721]]

 These were the hyperparameters for the model:
 Fitting RandomizedSearchCV for Temporal Set...
Fitting 5 folds for each of 50 candidates, totalling 250 fits
Best Parameters (Temporal Set): {'subsample': 0.4, 'n_estimators': 200, 'min_child_weight': 3, 'max_depth': 7, 'learning_rate': 0.01, 'gamma': 0, 'colsample_bytree': 0.4}
Best F1 Score (Temporal Set): 0.6594620380425328

"""