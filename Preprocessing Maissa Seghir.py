# AFTER CLEANING RUN THIS FILE
#Preprocessing the features so that the decision tree will be able to correctly process the data
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
from imblearn.over_sampling import SMOTENC

# Reading cleaned dataset
df = pd.read_csv('cleaned_data.csv')
print(df.head())
print(df.info())
print(df.describe())

# Data split into labeled and unlabeled data
# Data split into labeled and unlabeled data
labeled_data = df[df['Target'].notna()].copy()
unlabeled_data = df[df['Target'].isna()].copy()

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

print(f"Missing data % in labeled data:")
display_missing_values(labeled_data, max_columns=None, max_rows=None)
print(f"Missing data % in unlabeled data:")
display_missing_values(unlabeled_data, max_columns=None, max_rows=None)

#Dropping rows with missing values
labeled_data = labeled_data.dropna()

#Doing some feature transformation
# Transform "Target" into a binary column in labeled data
labeled_data['Target_binary'] = np.where(labeled_data['Contract_duur'] > 3, 1, 0)

#Removing the numbers from the street 
labeled_data['Straat'] = labeled_data['Straat'].str.replace(r'\d+', '', regex=True)
labeled_data['Straat'] = labeled_data['Straat'].str.strip()
print(labeled_data['Straat'].head(20))

#Removing letters from postal code, this loses some of its granularity but it's still more precise then a city name
labeled_data['Postcode_cijfers'] = labeled_data['Postcode'].str.replace(r'[A-Za-z]', '', regex=True)
labeled_data = labeled_data.drop(columns=["Postcode"])

# Convert columns to datetime using .loc to avoid SettingWithCopyWarning
labeled_data.loc[:, 'Construction'] = pd.to_datetime(labeled_data['Year of construction'])
labeled_data.loc[:, 'Demolition'] = pd.to_datetime(labeled_data['Year of demolition'])
labeled_data.loc[:, 'VABI_finished'] = pd.to_datetime(labeled_data['Amfelddatum_VABI'])
labeled_data.loc[:, 'Contract_starting'] = pd.to_datetime(labeled_data['Ingangsdatum_contract'])

# Extract year from all datetime columns using .loc to set new columns
for col in labeled_data.select_dtypes(include=['datetime']).columns:
    labeled_data.loc[:, f'{col}_year'] = labeled_data[col].dt.year

# Drop the original columns with .loc
labeled_data.drop(columns=['Year of construction', 'Year of demolition', 'Amfelddatum_VABI', 'Ingangsdatum_contract'], inplace=True)



# TEMPORAL SPLIT AND RANDOM SPLIT


# Perform a random split to determine the training set size for an 80/20 split
train_data_rand, test_data_rand = train_test_split(labeled_data, test_size=0.2, random_state=777)

# Get the size of the training set from the random split
train_size = train_data_rand.shape[0]

# Sort the labeled data by 'Huis_leeftijd' in descending order (oldest houses first)
labeled_data_sorted = labeled_data.sort_values(by='Huis_leeftijd', ascending=False)

# Select the top 'train_size' rows for the temporal training set
train_data_temp = labeled_data_sorted.iloc[:train_size]

# The remaining rows after the training set will form the test set
test_data_temp = labeled_data_sorted.iloc[train_size:]

# Display the sizes to verify the split
print("Training set size (temporal):", train_data_temp.shape[0])
print("Testing set size (temporal):", test_data_temp.shape[0])

# Check the sizes of the splits
print("Training set size (temporal):", train_data_temp.shape[0])
print("Testing set size (temporal):", test_data_temp.shape[0])
print("Total size:", labeled_data.shape[0])

#CHECK FOR TARGET CLASS BALANCE IN BOTH SETS:
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
    df = df.sort_values(['VIBDRO_Huurobject_id', 'Construction_year'])
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
columns_to_drop = ['Target','Ontvangstdatum_opzegging','Einddatum_contract','Contract_duur','Land','Huurobject', 'Energie_index', 
    'Marktwaarde', 'Totaal kamers', 'Totale punten (afgerond)', 
    'Totale punten (onafgerond)', 'WOZ waarde', 'WOZ waarde (WWS)', 'WOZ waarde per m2', 
    'WOZ waarde per m2 (WWS)', 'Woonkamer', 'Markthuur', 'Maximaal_redelijke_huur', 'Streefhuur', 
    'Year of demolition flag', 'Energielabel_encoded', 'EP2_waarde flag', 'Ontvangstdatum_opzegging flag', 
    'Reden_opzegging flag', 'Demolition_year', 'avg_contract_duration_per_property', 
    'avg_contract_duration_per_property_type', 'avg_contract_duration_per_complex', 
    'avg_contract_duration_per_city', 'avg_contract_duration_per_region', 'rolling_mean_property_value', 
    'Aantal_slaapkamers', 'Aardgasloze_woning','Geen_deelname_energieproject','Contractnummer','VIBDRO_Huurobject_id']  #Also including data thats 100% correlated to target variable, such as contract_end_date
#Also removed all the ID's
train_data_temp_with_features = train_data_temp_with_features.drop(columns=columns_to_drop)
train_data_rand_with_features = train_data_rand_with_features.drop(columns=columns_to_drop)



# Prepare features and target variable
X_temp = train_data_temp_with_features.drop(columns=['Target_binary'])
y_temp = train_data_temp_with_features['Target_binary']

X_rand = train_data_rand_with_features.drop(columns=['Target_binary'])
y_rand = train_data_rand_with_features['Target_binary']

#Going to do categorical smote

# Get categorical feature indices temp
categorical_indices_temp = [i for i, col in enumerate(X_temp.columns) if X_temp[col].dtype == 'object']
print("Indices of categorical features in X_temp:", categorical_indices_temp)

# Get categorical feature indices rand
categorical_indices_rand = [i for i, col in enumerate(X_rand.columns) if X_rand[col].dtype == 'object']
print("Indices of categorical features in X_rand:", categorical_indices_rand)

#Still get issues with some datetype columns, only extracting years from it, i know i did this earlier but its still giving me bugs:
# Convert datetime columns to year in X_temp
for col in X_temp.select_dtypes(include=['datetime64']):
    X_temp[col] = X_temp[col].dt.year

# Convert datetime columns to year in X_rand
for col in X_rand.select_dtypes(include=['datetime64']):
    X_rand[col] = X_rand[col].dt.year

# Check the number of rows in X_temp
num_rows_temp = X_temp.shape[0]  # or len(X_temp)
print(f"Number of rows in X_temp before SMOTENC: {num_rows_temp}")

# Check the number of rows in X_rand
num_rows_rand = X_rand.shape[0]  # or len(X_rand)
print(f"Number of rows in X_rand before SMOTENC: {num_rows_rand}")

"""#The categorical feautre indices for SMOTENC
categorical_features_indices = [1, 4, 5, 7, 8, 9, 10, 11, 31, 32, 33, 53, 61]

# Apply SMOTENC to the temporal training set
smote_nc_temp = SMOTENC(categorical_features=categorical_features_indices, random_state=777)
X_temp_balanced, y_temp_balanced = smote_nc_temp.fit_resample(X_temp, y_temp)

# Apply SMOTENC to the random training set
smote_nc_rand = SMOTENC(categorical_features=categorical_features_indices, random_state=777)
X_rand_balanced, y_rand_balanced = smote_nc_rand.fit_resample(X_rand, y_rand)

# Check the class distribution after SMOTENC
print("Class distribution in Temporal Training Set after SMOTENC:")
print(y_temp_balanced.value_counts())

print("Class distribution in Random Training Set after SMOTENC:")
print(y_rand_balanced.value_counts()) 

# Save the balanced temporal split for the next step
X_temp_balanced.to_csv('X_temp_balanced.csv', index=False)
y_temp_balanced.to_csv('y_temp_balanced.csv', index=False)

# Save the balanced random split for the next step
X_rand_balanced.to_csv('X_rand_balanced.csv', index=False)
y_rand_balanced.to_csv('y_rand_balanced.csv', index=False)"""
