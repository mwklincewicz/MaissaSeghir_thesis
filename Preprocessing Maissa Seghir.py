#AFTER CLEANING RUN THIS FILE

#Importing libraries 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
import re

#reading cleaned dataset 
df =pd.read_csv('cleaned_data.csv')
print(df.head())
print(df.info())
print(df.describe())

#Data split into labeled and unlabeled data
labeled_data = df[df['Target'].notna()]   
unlabeled_data = df[df['Target'].isna()]  

# Save the labeled and unlabeled data to separate CSV files 
labeled_data.to_csv('labeled_data.csv', index=False)
unlabeled_data.to_csv('unlabeled_data.csv', index=False)

#Checking for differences between missing data % in labelled and unlabelled data to check if i should do more missing data imputation 
def display_missing_values(df, max_columns=None, max_rows=None):
    pd.set_option('display.max_columns', max_columns)
    pd.set_option('display.max_rows', max_rows)
    
 
    missing_value_percentages = df.isnull().mean() * 100
    missing_value_percentages = missing_value_percentages.sort_values(ascending=False)
    
    print(missing_value_percentages)


print(f"Missing data % in labeled data:")
display_missing_values(labeled_data, max_columns=None, max_rows=None)
print(f"Missing data % in unlabeled data:")
display_missing_values(unlabeled_data, max_columns=None, max_rows=None)

#Labeled data has more missing % than unlabeled data, likeely because labeled data has older properties and thus worse registration. 