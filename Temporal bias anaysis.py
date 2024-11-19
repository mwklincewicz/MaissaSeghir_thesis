#Writing this code after i ran all my models 
#RUN BEFORE CLEANING AND AFTER PREPROCESSING

#loading data
import pandas as pd

df = pd.read_csv('cleaned_data.csv')
print(df.head())
print(df.info())
print(df.describe())

#I want to see how i can mitigate temporal bias, my current models are too reliant on temporal features
#I want to see how much of my data is before 2010, since most dt start of with this cut-off 

# column conversion
df['Ingangsdatum_contract'] = pd.to_datetime(df['Ingangsdatum_contract'], errors='coerce')

# Extract the year
df['contract_year'] = df['Ingangsdatum_contract'].dt.year

# Calculate the percentage of contracts after 2010
total_contracts = len(df)
contracts_after_2010 = len(df[df['contract_year'] >= 2010])
percentage_after_2010 = (contracts_after_2010 / total_contracts) * 100

print(f"Percentage of contracts after 2010: {percentage_after_2010:.2f}%")

#70% of the data are contracts from AFTER 2010. 
#Which is around 50k rows

# Filter for contracts after 2010 and where 'Contract_duur' is not -1 to see how much of this data is labeled and how much of it is unlabeled
valid_contracts = df[(df['contract_year'] >= 2010) & (df['Contract_duur'] != -1)]

# Calculate the percentage of labeled data after 2010
total_contracts = len(df)
percentage_valid_after_2010 = (len(valid_contracts) / total_contracts) * 100

print(f"Percentage of contracts after 2010 with 'Contract_duur' not equal to -1: {percentage_valid_after_2010:.2f}%")

#Labaled data after 2010 is 38%
#Which is around 27k rows

# Filter for contracts after 2010 where 'Contract_duur' is -1
contracts_after_2010_negative_duration = df[(df['contract_year'] >= 2010) & (df['Contract_duur'] == -1)]

# Calculate the percentage
total_contracts = len(df)
percentage_negative_after_2010 = (len(contracts_after_2010_negative_duration) / total_contracts) * 100

print(f"Percentage of contracts after 2010 with 'Contract_duur' equal to -1: {percentage_negative_after_2010:.2f}%")

#Unlabeled data after 2010 is 32%, which is which is around 23k rows

#Lets split the data
# Split the dataset into two based on the 'contract_year'
df_before_2010 = df[df['contract_year'] < 2010]
df_after_2010 = df[df['contract_year'] >= 2010]

# Check the sizes of the new DataFrames
print(f"Number of contracts before 2010: {len(df_before_2010)}")
print(f"Number of contracts after 2010: {len(df_after_2010)}")

# Calculate percentages
total_contracts = len(df)
percentage_before_2010 = (len(df_before_2010) / total_contracts) * 100
percentage_after_2010 = (len(df_after_2010) / total_contracts) * 100

print(f"Percentage of contracts before 2010: {percentage_before_2010:.2f}%")
print(f"Percentage of contracts after 2010: {percentage_after_2010:.2f}%")


#LETS CHECK THE CLASS BALANCE
#Important for later, i did smote first but didnt do much, if there is a class imbalance i might need undersampling (random sampling majority class)
# Check the class balance for 'Target' in both DataFrames
print("Class Balance for Contracts Before 2010:")

# Class balance before 2010
class_balance_before_2010 = df_before_2010['Target'].value_counts(normalize=True) * 100
print(class_balance_before_2010)

"""Class Balance for Contracts Before 2010:
>3     98.511205
<=3     1.488795"""

print("\nClass Balance for Contracts After 2010:")

# Class balance after 2010
class_balance_after_2010 = df_after_2010['Target'].value_counts(normalize=True) * 100
print(class_balance_after_2010)

"""Class Balance for Contracts After 2010:
<=3    57.762834
>3     42.237166"""

# Check the counts for each class in both DataFrames
counts_before_2010 = df_before_2010['Target'].value_counts()
counts_after_2010 = df_after_2010['Target'].value_counts()

print(f"\nCounts of each class in contracts before 2010:\n{counts_before_2010}")
print(f"\nCounts of each class in contracts after 2010:\n{counts_after_2010}")

# Writing the contracts after 2010 data to a new CSV file
df_after_2010.to_csv('contracts_after_2010.csv', index=False)

print("Data after 2010 has been saved to 'contracts_after_2010.csv'")