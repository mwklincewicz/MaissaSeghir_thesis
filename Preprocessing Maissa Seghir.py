# AFTER CLEANING RUN THIS FILE

# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
import re
from sklearn.model_selection import train_test_split

# Reading cleaned dataset
df = pd.read_csv('cleaned_data.csv')
print(df.head())
print(df.info())
print(df.describe())

# Data split into labeled and unlabeled data
labeled_data = df[df['Target'].notna()]
unlabeled_data = df[df['Target'].isna()]

# Save the labeled and unlabeled data to separate CSV files
labeled_data.to_csv('labeled_data.csv', index=False)
unlabeled_data.to_csv('unlabeled_data.csv', index=False)

# Checking for differences between missing data % in labelled and unlabelled data to check if i should do more missing data imputation
def display_missing_values(df, max_columns=None, max_rows=None):
    pd.set_option('display.max_columns', max_columns)
    pd.set_option('display.max_rows', max_rows)

    missing_value_percentages = df.isnull().mean() * 100
    missing_value_percentages = missing_value_percentages.sort_values(ascending=False)
    
    print(missing_value_percentages)

# I dont want to run this every time
# print(f"Missing data % in labeled data:")
# display_missing_values(labeled_data, max_columns=None, max_rows=None)
# print(f"Missing data % in unlabeled data:")
# display_missing_values(unlabeled_data, max_columns=None, max_rows=None)

# Labeled data has more missing % than unlabeled data, likely because labeled data has older properties and thus worse registration.
# Im just going to drop the rows with missing values for the column EP2_waarde since i dont know how else to impute it.
# Highest missing % is under 5 percent so im leaving the dataset like this as is.
labeled_data = labeled_data.dropna(subset=['EP2_waarde'])
unlabeled_data = unlabeled_data.dropna(subset=['EP2_waarde'])

# I dont want to run this every time
# print(f"Missing data % in labeled data after column drop:")
# display_missing_values(labeled_data, max_columns=None, max_rows=None)
# print(f"Missing data % in unlabeled data after column drop:")
# display_missing_values(unlabeled_data, max_columns=None, max_rows=None)

# TEMPORAL SPLIT

# I want an 80% training data and 20% test data, so calculating the 80th percentile of house ages:
# This way i get a fair 80/20 split
house_age_cutoff = labeled_data['Huis_leeftijd'].quantile(0.80)

# Putting 80% into training en 20% into testing based on temporal data of the house age
train_data_temp = labeled_data[labeled_data['Huis_leeftijd'] > house_age_cutoff]  # Training set with only older houses (80%)
test_data_temp = labeled_data[labeled_data['Huis_leeftijd'] <= house_age_cutoff]  # Testing set with only newer houses (20%)

# A random split on labeled data for a comparison in the experiment
train_data_rand, test_data_rand = train_test_split(labeled_data, test_size=0.2, random_state=777)  # randomstate 777 for luck!

# FEATURE ENGINEERING FOR TRAINING DATA
# IDEAS 
# Geo features
# Average contract duration per Property
# Average contract duration per Complex
# Average contract duration per city
# Average contract duration per postal code
# Average contract duration per Region

# Property features
# Average contract duration per property type
# Average contract duration per amount of bedrooms
# Rolling mean for size of property
# Rolling means for values of the property

# Lets start with these and add more if there is low correlation

def feature_engineering(df):
    features = pd.DataFrame(index=df.index)
    
    # Average Contract Duration per Property
    features['avg_contract_duration_per_property'] = df.groupby('VIBDRO_Huurobject_id')['Contract_duur'].transform('mean')
    
    # Average Contract Duration per Property Type
    features['avg_contract_duration_per_property_type'] = df.groupby('Omschrijving_Vastgoed')['Contract_duur'].transform('mean')
    
    # Average Contract Duration per Complex
    features['avg_contract_duration_per_complex'] = df.groupby('complexnummer')['Contract_duur'].transform('mean')
    
    # Average Contract Duration per City
    features['avg_contract_duration_per_city'] = df.groupby('Gemeente')['Contract_duur'].transform('mean')
    
    # Average Contract Duration per Region
    features['avg_contract_duration_per_region'] = df.groupby('regio')['Contract_duur'].transform('mean')

    # Rolling Mean for Property Value
    df = df.sort_values(['VIBDRO_Huurobject_id', 'Year of construction'])
    features['rolling_mean_property_value'] = df.groupby('VIBDRO_Huurobject_id')['WOZ waarde'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    
    return pd.concat([df.reset_index(drop=True), features.reset_index(drop=True)], axis=1)

# Apply feature engineering
train_data_temp_with_features = feature_engineering(train_data_temp)
train_data_rand_with_features = feature_engineering(train_data_rand)

# Function to plot correlation matrix, excluding categorical data that does not correlate numerically
def plot_correlation_matrix(df, title):
    plt.figure(figsize=(10, 8))
    
    # Selecting only numeric columns for correlation calculation in the plot
    numeric_df = df.select_dtypes(include=['number'])
    corr_matrix = numeric_df.corr()
    
    # Focus only on the correlation with target variable
    if 'Target' in corr_matrix.columns:
        target_corr = corr_matrix[['Target']].sort_values(by='Target', ascending=False)
        
        # Plot the heatmap
        sns.heatmap(target_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title(title)
        plt.show()
    else:
        print("Warning: 'Target' not found in the correlation matrix.")

# Plot correlation matrix for temporal and random training sets
print("Correlation Matrix for Temporal Training Set")
plot_correlation_matrix(train_data_temp_with_features, "Correlation with Target - Temporal Training Set")

print("Correlation Matrix for Random Training Set")
plot_correlation_matrix(train_data_rand_with_features, "Correlation with Target - Random Training Set")
