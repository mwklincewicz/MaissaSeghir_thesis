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
from scipy.stats import chi2_contingency
from matplotlib_venn import venn2
from imblearn.over_sampling import SMOTE

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


# Transform "Target" into a binary column in labeled data
labeled_data['Target_binary'] = np.where(labeled_data['Contract_duur'] > 3, 1, 0)

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

#CHECK FOR TARGET VARIABLE BALANCE IN BOTH SETS:
#Plot class balance for both classes
def check_class_balance(data, target_variable, title):
    class_counts = data[target_variable].value_counts()
    print(f"Class distribution for {title}:")
    print(class_counts)

    # Plotting class balance
    plt.figure(figsize=(8, 5))
    sns.barplot(x=class_counts.index.astype(str), y=class_counts.values)
    plt.title(f'Class Balance for {title}')
    plt.xlabel('Classes')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.show()

# Checking class balance for the temporal training split
check_class_balance(train_data_temp, 'Target_binary', 'Temporal Training Split')

# Checking class balance for the random training split
check_class_balance(train_data_rand, 'Target_binary', 'Random Training Split')


#There is a class imbalance in both instances where 1 is overrepresentend and 0 is underrepresented. Im doing SMOTE after feature engineering to 
#Make sure classes are balanced in training data 


# FEATURE ENGINEERING FOR TRAINING DATA
# IDEAS 
# Geo features
# Average contract duration per Property
# Average contract duration per Complex
# Average contract duration per city

# Average contract duration per Region

# Property features
# Average contract duration per property type
# amount of bedrooms
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

    #Amount of bedrooms in a house
    features['Aantal_slaapkamers'] = (
        (df['1e Slaapkamer'] > 0).astype(int) +
        (df['2e Slaapkamer'] > 0).astype(int) +
        (df['3e Slaapkamer'] > 0).astype(int)
    )
    
    def assign_huurklasse(markthuur):
        if pd.isna(markthuur):
            return 'Onbekend'
        elif markthuur <= 454.47:  # tot kwaliteitskortingsgrens
            return 'Goedkoop'
        elif markthuur <= 650.43:  # kwaliteitskortingsgrens - eerste aftoppingsgrens
            return 'klaliteitskorting Betaalbaar'
        elif markthuur <= 697.08:  # eerste aftoppingsgrens - tweede aftoppingsgrens
            return '1e aftoppingsgrens Betaalbaar'
        elif markthuur <= 879.66:  # tweede aftoppingsgrens - huurtoeslaggrens/DAEB-grens/liberalisatiegrens
            return 'Sociale hogere huur'
        else:
            return 'Vrije huur'

    features['Huurklasse'] = df['Markthuur'].apply(assign_huurklasse)
    
    df_with_features = pd.concat([df.reset_index(drop=True), features.reset_index(drop=True)], axis=1)

    return df_with_features

# Apply feature engineering
train_data_temp_with_features = feature_engineering(train_data_temp)
train_data_rand_with_features = feature_engineering(train_data_rand)

# Function to plot correlation matrix, excluding categorical data that does not correlate numerically
def plot_correlation_matrix(df, title):
    plt.figure(figsize=(12, 10))
    
    # Selecting only numeric columns for correlation calculation in the plot
    numeric_df = df.select_dtypes(include=['number'])
    corr_matrix = numeric_df.corr()
    
    # Focus only on the correlation with target variable
    if 'Target_binary' in corr_matrix.columns:
        target_corr = corr_matrix[['Target_binary']].sort_values(by='Target_binary', ascending=False)
        
        # Plot the heatmap
        sns.heatmap(target_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title(title)
        plt.show()
    else:
        print("'Target_binary' not found in the correlation matrix.")

# Plot correlation matrix for temporal and random training sets
print("Correlation Matrix for Temporal Training Set") #I dont want to run this every time
plot_correlation_matrix(train_data_temp_with_features, "Correlation with Target - Temporal Training Set") #I dont want to run this every time

print("Correlation Matrix for Random Training Set")
plot_correlation_matrix(train_data_rand_with_features, "Correlation with Target - Random Training Set")

#House age high higest corr in the random split but not high corr in the temporal split
#Which means the temporal split managed to get out some of the temporal bias in the data
#Numerical variables arent very high in correlation, with highest corr features being around 20%, feature engineering resulted in weak features. 
#Rooms gave good correlation and WOZ-value, also the neighbourhood scored well. Maybe make more features around the house (e.g., amount of bedrooms)
#For the sake of consistency i want to drop the same columns in both training sets, so the experiment gauges the difference in splits
#Instead of the different in feature sets
#So im doing a comparison using a venn diagram
# Function to get columns with low correlation
def get_low_correlation_columns(df, threshold=0.1): #Threshold is 10% correlation 
    corr_matrix = df.corr()
    if 'Target_binary' in corr_matrix.columns:
        low_corr_cols = corr_matrix['Target_binary'][abs(corr_matrix['Target_binary']) < threshold].index.tolist()
        return low_corr_cols
    else:
        print("'Target_binary' not found in the correlation matrix.")
        return []

# Get low correlation columns for both random and temporal split training datasets
low_corr_temp = get_low_correlation_columns(train_data_temp_with_features)
low_corr_rand = get_low_correlation_columns(train_data_rand_with_features)

# Print the results
print(f"Low Correlation Columns (Temporal Split): {low_corr_temp}")
print(f"Low Correlation Columns (Random Split): {low_corr_rand}")

# Create a Venn diagram for visualization
plt.figure(figsize=(10, 6))
venn_labels = {'10': len(set(low_corr_temp)), '01': len(set(low_corr_rand)), '11': len(set(low_corr_temp) & set(low_corr_rand))}
venn2(subsets=venn_labels, set_labels=('Temporal Split', 'Random Split'))
plt.title("Venn Diagram of Low Correlation Columns (< 10% Correlation)")
plt.show()

#THere is some intersection
#Random split overal has higher correlation to features, which i think is due to the temporal bias in the data
#Which also means the temporal split did filter out some of the temporal bias, since temporal features are low % in the temporal split
#Columns below 10% correlation will be deleted to prevent overfitting

# Now doing the chi2 test for categorical features
# Define the categorical features in the dataset
categorical_features = [
    'Woning_type', 
    'Energielabel', 
    'Aardgasloze_woning',
    'Geen_deelname_energieproject',
    'Contractsoort', 
    'Reden_opzegging',
    'Omschrijving_Vastgoed',
    'Eengezins_Meergezins',
    'VERA_Type',
    'Straat',
    'Gemeente',
    'regio',
    'Huurklasse'
]

# Store Chi-Squared test results separately
chi2_results_temp = {}
chi2_results_rand = {}

# Function to perform Chi-Squared test for a given DataFrame and target variable 
def perform_chi_squared_test(data, target_variable, results_dict):
    for feature in categorical_features:
        if feature in data.columns:  
            contingency_table = pd.crosstab(data[feature], data[target_variable])
            
            # Perform the Chi-Squared test
            chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
            
            # Store the results in the specified results dictionary
            results_dict[feature] = {
                'chi2_statistic': chi2_stat,
                'p_value': p_value,
                'degrees_of_freedom': dof,
                'expected_frequencies': expected
            }

# Perform Chi-Squared test on the temporal training dataset
print("Chi-Squared Test Results for Temporal Training Set")
perform_chi_squared_test(train_data_temp_with_features, 'Target_binary', chi2_results_temp)

# Convert results to a DataFrame for better readability
chi2_results_temp_df = pd.DataFrame(chi2_results_temp).T
print(chi2_results_temp_df)

# Perform Chi-Squared test on the random training dataset
print("Chi-Squared Test Results for Random Training Set")
perform_chi_squared_test(train_data_rand_with_features, 'Target_binary', chi2_results_rand)

# Convert results to a DataFrame for better readability
chi2_results_rand_df = pd.DataFrame(chi2_results_rand).T
print(chi2_results_rand_df)

# Function to interpret the Chi-Squared test results based on alpha == 0.05 to see if there is a significant relation or not.
def interpret_chi2_results(chi2_results_df, alpha=0.05):
    for feature, results in chi2_results_df.iterrows():
        chi2_stat = results['chi2_statistic']
        p_value = results['p_value']
        
        print(f"Feature: {feature}")
        print(f"Chi-Squared Statistic: {chi2_stat:.4f}")
        print(f"P-Value: {p_value:.4f}")
        
        if p_value < alpha:
            print("Conclusion: Reject the null hypothesis. There is a significant association between this feature and the target variable.")
        else:
            print("Conclusion: Fail to reject the null hypothesis. No significant association found.")
        print("-" * 50)

# Interpret Chi-Squared results for temporal training dataset
print("Interpreting Chi-Squared Test Results for Temporal Training Set")
interpret_chi2_results(chi2_results_temp_df)

# Interpret Chi-Squared results for random training dataset
print("Interpreting Chi-Squared Test Results for Random Training Set")
interpret_chi2_results(chi2_results_rand_df)

#Most categorical features are highly correlated with the target variable, That is good
#Going to drop low correlation columns 


#Going to drop low corr for temporal data and random data
#Dropping the same columns in temporal split and random split for the sake of Consistent preprocessing, 
# so that any differences in model performance can be attributed to the splits themselves rather than discrepancies in the feature set
columns_to_drop = ['Target','Ontvangstdatum_opzegging','Einddatum_contract','Contract_duur','Land',
'Aardgasloze_woning','Geen_deelname_energieproject','VERA_Type','Straat','Huurklasse','Energie_index',
'Badkamer/doucheruimte 1', 'Toilet (Sanitair 1)', 'Totale punten (afgerond)', 'Totale punten (onafgerond)',
'WOZ waarde per m2', 'WOZ waarde per m2 (WWS)', 'Woonkamer', 'Maximaal_redelijke_huur', 'Streefhuur',
'Verwarmde overige ruimten flag','Year of demolition flag', 'Energielabel_encoded', 'EP2_waarde flag',
'Ontvangstdatum_opzegging flag', 'Reden_opzegging flag', 'avg_contract_duration_per_property',
'avg_contract_duration_per_property_type', 'avg_contract_duration_per_complex','avg_contract_duration_per_city',
'avg_contract_duration_per_region','rolling_mean_property_value','Aantal_slaapkamers' ]  #Also including data thats 100% correlated to target variable, such as contract_end_date
train_data_temp_with_features = train_data_temp_with_features.drop(columns=columns_to_drop)
train_data_rand_with_features = train_data_rand_with_features.drop(columns=columns_to_drop)

# Prepare features and target variable
X_temp = train_data_temp_with_features.drop(columns=['Target_binary'])
y_temp = train_data_temp_with_features['Target_binary']

X_rand = train_data_rand_with_features.drop(columns=['Target_binary'])
y_rand = train_data_rand_with_features['Target_binary']

#I will still need to encode categorical variables if i want to use SMOTE... 
#SO despite trees not really needing encoding i will still encode categorical variables so i can use SMOTE effectively



"""# Apply SMOTE to the temporal training set
smote = SMOTE(random_state=777)
X_temp_balanced, y_temp_balanced = smote.fit_resample(X_temp, y_temp)

# Apply SMOTE to the random training set
X_rand_balanced, y_rand_balanced = smote.fit_resample(X_rand, y_rand)

# Check the class distribution after SMOTE
print("Class distribution in Temporal Training Set after SMOTE:")
print(y_temp_balanced.value_counts())

print("Class distribution in Random Training Set after SMOTE:")
print(y_rand_balanced.value_counts())"""
