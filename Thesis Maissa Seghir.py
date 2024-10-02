#Importing nessecary libraries
import pandas as pd 
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






