#Importing nessecary libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

#Open the file and check df
df =pd.read_csv('thesis DSS.csv')
print(df.head())
print(df.info())
print(df.describe())

#Check target variable statistics, checking for differences between mean and median, checking distribution
print(df['Contract_duur'].describe())
print(df['Contract_duur'].median())

#Botplot for visualisation 
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)  
plt.boxplot(df['Contract_duur'].dropna())
plt.title('Boxplot of target variable')
plt.ylabel('Contract duration')
#plt.show() #remove hashtag if you want to see this plot, I personally dont want to see this plot every time i run the code

#Historgram for further visualisation 
target_variable= df['Contract_duur'].dropna()

plt.figure(figsize=(10, 5))
sns.histplot(target_variable, kde=False, bins=100, color='grey', stat='density')

plt.title('Distribution of contract durations')
plt.xlabel('Contract duration')
plt.ylabel('Density')
#plt.show() #remove hashtag if you want to see this plot, I personally dont want to see this plot every time i run the code 

#I want to check the house age to see if there is a temporal bias, more new houses = shorter contracts
#first i need to convert dates and clean them to remove 9999 dates, which is something databricks does automatically when extracting from sap. 
current_year = datetime.now().year
df['Year of construction'] = pd.to_datetime(df['Year of construction'])
df['Year of demolition'] = df['Year of demolition'].replace(['9999-12-31'], [None])
df['Year of demolition'] = pd.to_datetime(df['Year of demolition'], errors='coerce')

def calculate_house_age(row):
    year_demolition = row['Year of demolition'].year if pd.notnull(row['Year of demolition']) else current_year
    return year_demolition - row['Year of construction'].year

df['Huis_leeftijd'] = df.apply(calculate_house_age, axis=1)

print(df['Huis_leeftijd'].describe())

#Now i want to plot the distribution of the house ages to see if there are many new houses or not
house_age= df['Huis_leeftijd'].dropna()

plt.figure(figsize=(10, 5))
sns.histplot(house_age, kde=False, bins=100, color='grey', stat='density')

plt.title('Distribution of house ages')
plt.xlabel('house age')
plt.ylabel('Density')
#plt.show()#remove hashtag if you want to see this plot, I personally dont want to see this plot every time i run the code

#check missing values
import pandas as pd

def display_missing_values(df, max_columns=None, max_rows=None):
    pd.set_option('display.max_columns', max_columns)
    pd.set_option('display.max_rows', max_rows)
    
 
    missing_value_percentages = df.isnull().mean() * 100
    missing_value_percentages = missing_value_percentages.sort_values(ascending=False)
    
    print(missing_value_percentages)

#print(display_missing_values(df, max_columns=None, max_rows=None)) #remove hashtag if you want to see this code. 


#drop duplicates and high %, except for columns that have missing columns for a reason (example, an empty value in year of demolition means the house is still standing)
#And drop duplicates
missing_value_percentages = df.isnull().mean() * 100
threshold = 85
df = df.loc[:, missing_value_percentages <= threshold]
drop = ['REkey_vicncn', 'ID_Huurovereenkomst', 'VIBDMEAS_Huurobject_id', 'VICDCONDCALC_ID', 'id_huurovereenkomst1', 'id_huurovereenkomst2']
df = df.drop(columns=[col for col in drop if col in df.columns])

#print(df.head()) #remove hashtag if you want to see this code. 

#Treating missing values by data imputation 
#A lot of conditional missing data, e.g. a parking spot is not going to have an energy label obviously, so we will need to treat every column type seperately

#Also, handling values that are empty but should be 0 (for example, an empty cell in the 3rd bedroom does not mean its unknown how big the bedroom is, it means there is no bedroom so it should be 0)
rooms = ["Zolder", "Verwarmde overige ruimten", "2e Slaapkamer", 
                      "Aparte douche/lavet+douche 1", "Bergruimte/schuur 1", 
                     "Verwarmde vertrekken", "Totaal overige ruimtes", 
                       "Keuken", "Badkamer/doucheruimte 1", 
                      "Totaal kamers", "Woonkamer", "Toilet (Sanitair 1)"]

real_rooms = [col for col in rooms if col in df.columns]

df[real_rooms] = df[real_rooms].fillna(0)

#handling values that are 0, but should be missing instead (example: WOZ-value cannot be 0, this would mean the house is free)
prices = ["WOZ-waarde", "WOZ waarde (WWS)", "Marktwaarde", "Leegwaarde", 
                           "Historische kostprijs", "WOZ waarde per m2", 
                           "WOZ waarde per m2 (WWS)", "Streefhuur", "Markthuur"]

real_prices = [col for col in prices if col in df.columns]

df[real_prices] = df[real_prices].replace(0, np.nan)


df.to_csv('cleaned_data.csv', index=False)

#Treating missing values by data imputation 


#Also, handling values that are empty but should be 0 (for example, an empty cell in the 3rd bedroom does not mean its unknown how big the bedroom is, it means there is no bedroom so it should be 0)
rooms = ["Zolder", "Verwarmde overige ruimten", "2e Slaapkamer", 
                    "Aparte douche/lavet+douche 1", "Bergruimte/schuur 1", "3e Slaapkamer","Wastafel/bidet/lavet/fontein 1",
                    "1e Slaapkamer",  
                    "Verwarmde vertrekken", "Totaal overige ruimtes", 
                    "Keuken", "Badkamer/doucheruimte 1", 
                    "Totaal kamers", "Woonkamer", "Toilet (Sanitair 1)"]

real_rooms = [col for col in rooms if col in df.columns]

#adding a placeholder of 0 for the non excisting rooms
df[real_rooms] = df[real_rooms].fillna(0)

#adding a binary flag column for every room, so we know which rooms dont exist in which house
for room in real_rooms:
    df[f'{room} flag'] = (df[room] == 0).astype(int)


#handling values that are 0, but should be missing instead (example: WOZ-value cannot be 0, this would mean the house is free)
prices = ["WOZ-waarde", "WOZ waarde (WWS)", "Marktwaarde", "Leegwaarde", 
                           "Historische kostprijs", "WOZ waarde per m2", 
                           "WOZ waarde per m2 (WWS)", "Streefhuur", "Markthuur"]

real_prices = [col for col in prices if col in df.columns]

df[real_prices] = df[real_prices].replace(0, np.nan)

#adding a binary flag as a new column, so the model can interpret houses that arent broken down or sold
df['Year of demolition flag'] = df['Year of demolition'].isnull().astype(int)

#adding a placeholder for year of demolition,  i removed it earlier for analytical purposes but i want it back, however 9999 is out of bounds so i have to keep it within bounds. 
df['Year of demolition'].fillna(pd.Timestamp('2100-12-31'), inplace=True)

#non residential properties dont have an energylabel, so ill add a placeholder in that column and a binary flag if its empty 
#before Energielabel is missing about 20% of data
df['Energielabel flag'] = df['Energielabel'].isnull().astype(int)

condition = df['Omschrijving_Vastgoed'].isin(['Woonwagen', 'Woonwagenstandplaats', 'Parkeerplaats auto','Parkeerplaats overdekt', 'Garage','Berging'])
df.loc[condition & df['Energielabel'].isna(), 'Energielabel'] = 'N/A'



df.to_csv('cleaned_data.csv', index=False)

cleaned_df =pd.read_csv('cleaned_data.csv')

print(display_missing_values(cleaned_df, max_columns=None, max_rows=None))


