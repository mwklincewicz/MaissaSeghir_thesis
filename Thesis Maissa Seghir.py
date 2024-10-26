#Importing nessecary libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
import re


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

#making my binary target variable
df['Target'] = df['Contract_duur'].apply(lambda x: '<=3' if x <= 3 else '>3' if pd.notnull(x) else '')

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
#Dropping all rows that have missing values in column "VIBDRO", this just means the contact doesnt have a property connected to it
#Idk why this happens but its only 0,2% and if its missing in vibro many other columns are also missing do these rows are kind of useless
df = df.dropna(subset=['VIBDRO_Huurobject_id'])

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
#some residental properties dont require an energylabel by law (this changed recently but the energylabels havent been added to the data yet)
#before Energielabel is missing about 20% of data
df['Energielabel'].replace('', np.nan, inplace=True)

condition = df['Omschrijving_Vastgoed'].isin(['Woonwagen', 'Woonwagenstandplaats', 
'Parkeerplaats auto','Parkeerplaats overdekt', 'Garage','Berging','Brede school',
'Cultuur ruimte','Dagbestedingsruimte','Horeca','Kantoorruimte','Hospice','Maatschappelijk werkruimte wijk-/buurtgericht',
'Opvangcentrum','Praktijkruimte','Psychische zorginstelling','Schoolgebouw','Verpleeghuis','Verstandelijk gehandicapten instelling'
'Welzijnswerkruimte wijk-/buurtgericht','Winkelruimte','Zorgsteunpunt','Wijksportvoorziening'])

df.loc[condition & df['Energielabel'].isna(), 'Energielabel'] = 'N.v.t.'
#now missing % in energielabel is 8,5%
#Add a binary flag for all "N.v.t." energylabels, these are the properties which arent meant to have a energielabel in the first place
#Or there is no legal requirement for an energylabel, such as for a school

df['Energielabel flag'] = (df['Energielabel'] == 'N.v.t.').astype(int)

#For the properties named as "kamer" it means this is a room, individual rooms do not have an energylabel, but the property itsself does
#So for every "Complex" (which is the overarching property in which multiple rooms are being rented) im looking up its energylabel in the Dutch Cadastre 
#which is a land registry 
#im doing this for all large complexes with multiple rooms
df['Energielabel'].replace('', np.nan, inplace=True)
df.loc[df['Energielabel'].isna() & (df['complexnummer'] == 1193.0), 'Energielabel'] = 'A'
df.loc[df['Energielabel'].isna() & (df['complexnummer'] == 683.0), 'Energielabel'] = 'A'
df.loc[df['Energielabel'].isna() & (df['complexnummer'] == 1256.0), 'Energielabel'] = 'A'
df.loc[df['Energielabel'].isna() & (df['complexnummer'] == 345.0), 'Energielabel'] = 'D'
df.loc[df['Energielabel'].isna() & (df['complexnummer'] == 7031.0), 'Energielabel'] = 'D'
df.loc[df['Energielabel'].isna() & (df['complexnummer'] == 344.0), 'Energielabel'] = 'B'
df.loc[df['Energielabel'].isna() & (df['complexnummer'] == 6.0), 'Energielabel'] = 'A'
df.loc[df['Energielabel'].isna() & (df['complexnummer'] == 1230.0), 'Energielabel'] = 'A+'
df.loc[df['Energielabel'].isna() & (df['complexnummer'] == 1020.0), 'Energielabel'] = 'A'
df.loc[df['Energielabel'].isna() & (df['complexnummer'] == 1122.0), 'Energielabel'] = 'A'
df.loc[df['Energielabel'].isna() & (df['complexnummer'] == 1131.0), 'Energielabel'] = 'B'
df.loc[df['Energielabel'].isna() & (df['complexnummer'] == 1191.0), 'Energielabel'] = 'A'
df.loc[df['Energielabel'].isna() & (df['complexnummer'] == 2018.0), 'Energielabel'] = 'A'
df.loc[df['Energielabel'].isna() & (df['complexnummer'] == 1257.0), 'Energielabel'] = 'N.v.t.'


#Complex 2082 has two different energylabels so i have to manually put in the adress

df.loc[df['Energielabel'].isna() & (df['Straat'] == 'Bergweg 8'), 'Energielabel'] = 'D'
df.loc[df['Energielabel'].isna() & (df['Straat'] == 'Bergweg 300'), 'Energielabel'] = 'G'


#energielabel is now missing around 5,8%, the rest of the houses without an energielabel mostly already sold or broken down. So this data is indead missing 
#and not conditionally missing, so for this I can do something like KNN imputation 
#using GeeksforGeeks code for this, source= https://www.geeksforgeeks.org/python-imputation-using-the-knnimputer/

#labelencoder because KNN works with numerical data, i havent found a good way to do it with categorial data so im just encoding it first
le = LabelEncoder()
df['Energielabel_encoded'] = le.fit_transform(df['Energielabel'].astype(str))

#Put back the NaN where they originally are in the column 
df.loc[df['Energielabel'].isna(), 'Energielabel_encoded'] = np.nan

#I want to use the columns house age and WOZ-value, since there are numerical and usually have high correlation with energie label, old houses and cheaper houses usually have lower energylabels then expensive and new houses
features = df[['WOZ waarde','Huis_leeftijd','Verhuurbaar vloeroppervlakte','Verwarmde vertrekken','Bruto vloeroppervlakte','Energielabel_encoded']]  

#Impute
imputer = KNNImputer(n_neighbors=5, weights="uniform")  # choose n=5 cause this is usually the default
imputed_data = imputer.fit_transform(features)

# Replace the imputed Energielabel values in the original df
df['Energielabel_encoded'] = np.where(df['Energielabel_encoded'].isna(), 
                                      imputed_data[:, 5], 
                                      df['Energielabel_encoded'])

# Decode the imputed numeric values back to their original categorical values, i dont really have to do this one cause the algoritmh doesnt care if its encoded or not
#But i will do it anyway to make the code more understandable. Trees work well with categories so its okay 
df['Energielabel'] = le.inverse_transform(df['Energielabel_encoded'].round().astype(int))

# Check changes
#print(df[['Energielabel', 'Energielabel_encoded']].head())  # remove hashtag if you want to see this code

#fixing empty columns in Afmelddatum_VABI, if there is no date it means this project did not include the property. So it isnt missing data.
#ill add a placeholder in the future and a binary flag.
df['Afmelddatum_VABI flag'] = df['Amfelddatum_VABI'].isnull().astype(int)
df['Amfelddatum_VABI'].fillna(pd.Timestamp('2100-12-31'), inplace=True)

#Doing the same for this feature
df['Ontvangstdatum_opzegging flag'] = df['Ontvangstdatum_opzegging'].isnull().astype(int)
df['Ontvangstdatum_opzegging'].fillna(pd.Timestamp('2100-12-31'), inplace=True)

#Also doing the same for this feature
df['Reden_opzegging'] = df['Reden_opzegging'].fillna('N.v.t.')
df['Reden_opzegging flag'] = (df['Reden_opzegging'] == 'N.v.t.').astype(int)

#The same for this  feature 
df['Contract_duur flag'] = df['Contract_duur'].isnull().astype(int)
df['Contract_duur'].fillna(-1, inplace=True)


#Going to do some conditional imputation for Omschrijving_Vastgoed, Eengezins_Meergezins, VERA_Type 
#If the "Contracttype" is "Woonwagen/standplaats" the Omschrijving vastgoed can only be Woonwagen on standplaats. If there are any rooms its woonwagen
#if no rooms omschrijving vastgoed is woonwagenstandplaats
# Conditional imputation for 'Omschrijving_Vastgoed' based on 'Contracttype' and room availability
def impute_omschrijving_vastgoed(row):
    if row['Contractsoort'] == 'Woonwagen/Standplaats':
        if (row['1e Slaapkamer'] > 0) or (row['2e Slaapkamer'] > 0) or (row['3e Slaapkamer'] > 0):
            return 'Woonwagen'
        else:
            return 'Woonwagenstandplaats'
    return row['Omschrijving_Vastgoed']  

# Apply the function 
df['Omschrijving_Vastgoed'] = df.apply(impute_omschrijving_vastgoed, axis=1)

#doing the same for Vera type and eengezins_meergezins
def impute_eengezins_meergezins_and_vera_type(row):
    # If 'Omschrijving_Vastgoed' is 'Woonwagenstandplaats'
    if row['Omschrijving_Vastgoed'] == 'Woonwagenstandplaats':
        row['Eengezins_Meergezins'] = 'Niet benoemd'
        row['VERA_Type'] = 'Overig'
    # If 'Omschrijving_Vastgoed' is 'Woonwagen'
    elif row['Omschrijving_Vastgoed'] == 'Woonwagen':
        row['Eengezins_Meergezins'] = 'Overig'
        row['VERA_Type'] = 'Woonruimte'
    return row

# Apply the function 
df = df.apply(impute_eengezins_meergezins_and_vera_type, axis=1)


#I got a file from work in which every complex has a "Omschrijving_vastgoed"
#I will imput "Omschrijving_vastgoed" from this data
#Dataset has another delimiter so i have to mention it else it wont run
#Make a column thats unique and the same to i can leftjoin 
#This code doesnt do anything to decrease missing values so im putting in """ """so it doesnt run everytime because it takes a long time to run
#If you want to run it remove the """ """
"""
df['Bedrijfscode'] = pd.to_numeric(df['Bedrijfscode'], errors='coerce').astype('Int64') 
df['complexnummer'] = pd.to_numeric(df['complexnummer'], errors='coerce').astype('Int64')
df['Huurobject'] = pd.to_numeric(df['Huurobject'], errors='coerce').astype('Int64')

df['VHE'] = 'HO ' + df['Bedrijfscode'].astype(str) + '/' + \
            df['complexnummer'].astype(str) + '/' + \
            df['Huurobject'].astype(str)

print(df['VHE'])

impute_df = pd.read_csv('bezitslijst per 02092024.csv', delimiter=';')

#Change to string to avoid issues
df['VHE'] = df['VHE'].astype(str)
impute_df['VHE nummer'] = impute_df['VHE nummer'].astype(str)

#merge on VHE
merged_df = df.merge(impute_df[['VHE nummer', 'VERA typering']], 
                     left_on='VHE', right_on='VHE nummer', how='left')

#impute
merged_df['Omschrijving_Vastgoed'] = merged_df.apply(
    lambda row: row['VERA typering'] if pd.isna(row['Omschrijving_Vastgoed']) else row['Omschrijving_Vastgoed'], axis=1
)

#Drop collumns i dont want
merged_df = merged_df.drop(columns=['VHE nummer', 'VERA typering'])

df = merged_df

"""

#This didnt seem to work, the missing values seem to be from houses that we currently no longer have. 
#I do have encoded Woning_type, so i will conditionally impute it based on that information 
#lets make a dictionary for encoding
dict = {
    1000: "Eengezinswoning",
    1010: "Appartement",
    1020: "Seniorenwoning",
    1030: "Woonzorgwoning",
    1040: "Serviceflatwoning",
    1050: "Verzorgingscentra",
    1060: "Begeleid wonen",
    1070: "Meergezinshuis",
    1080: "Maisonette",
    1090: "Kamer",
    1100: "Logeerkamer",
    1110: "Chalet",
    1120: "Woonwagen",
    1130: "Standpl. woonwagen",
    1140: "Garage",
    1150: "Parkeerplaats",
    1160: "Overd. parkeerplaats",
    1170: "Bergruimte",
    1180: "Bedrijfsruimte",
    1190: "Kantoor",
    1200: "Winkel",
    1210: "Praktijk",
    1220: "Peuterzaal",
    1230: "Kinderdagverblijf",
    1240: "Ontmoetingscentrum",
    1250: "Wijkgebouw",
    1260: "Beheer derden woning",
    1261: "Onderhoud particulieren",
    1262: "Onderhoud personeel"
}


df['Woning_type'] = df['Woning_type'].replace(dict)


#print(df[['Woning_type']])

#Lets impute

# Define the mapping for imputing Omschrijving_Vastgoed based on Woning_Type
impute_dict = {
    "Bergruimte": "Berging",
    "Appartement": "Appartement",
    "Seniorenwoning": "Seniorenwoning",
    "Woonzorgwoning": "Verzorgingshuis",
    "Serviceflatwoning": "Serviceflatwoning",
    "Verzorgingscentra": "Verpleeghuis",
    "Begeleid wonen": "Begeleid wonen",
    "Meergezinshuis": "Meergezinshuis",
    "Maisonette": "Maisonette",
    "Kamer": "Kamer",
    "Logeerkamer": "Kamer",
    "Chalet": "Tijdelijke woning",
    "Woonwagen": "Woonwagen",
    "Standpl. woonwagen": "Woonwagenstandplaats",
    "Garage": "Garage",
    "Parkeerplaats": "Parkeerplaats auto",
    "Overd. parkeerplaats": "Parkeerplaats overdekt",
    "Bedrijfsruimte": "Bedrijfsruimte",
    "Kantoor": "Kantoorruimte",
    "Winkel": "Winkelruimte",
    "Praktijk": "Praktijkruimte",
    "Peuterzaal": "Schoolgebouw",
    "Kinderdagverblijf": "Schoolgebouw",
    "Ontmoetingscentrum": "Welzijnswerkruimte wijk-/buurtgericht",
    "Wijkgebouw": "Welzijnswerkruimte wijk-/buurtgericht",
}

# conditional imputation 
df['Omschrijving_Vastgoed'] = df.apply(
    lambda row: impute_dict.get(row['Woning_type'], row['Omschrijving_Vastgoed']) 
    if pd.isna(row['Omschrijving_Vastgoed']) else row['Omschrijving_Vastgoed'],
    axis=1
)

#Omschrijving_Vastgoed went from 17% missing to 5% missing

#now I can imput Eengezins_Meergezins and VERA_Type based on Omschrijving_Vastgoed and Woning_type
#if woning_type is eengezins impute that into eengezins_meergezins, doing the same for meergezins proprety (So if one family is living there or more)
#e.g. a flat is meergezins, because multiple groups of people reside there, a normal house is an eengezinswoning, because one family resides
def impute_eengezins_meergezins(row):
    if pd.isna(row['Eengezins_Meergezins']):
        if row['Woning_type'] == "Eengezinswoning":
            return "Eengezinswoning"
        elif row['Woning_type'] == "Meergezinswoning":
            return "Meergezinswoning"
    return row['Eengezins_Meergezins']

#Apply conditional imputation
df['Eengezins_Meergezins'] = df.apply(impute_eengezins_meergezins, axis=1)

#Eengezins_Meergezins went from 17% to 12%
#Using Woningtype for conditionally imputing eengezins_meergezins further
impute_dict_eengezins_meergezins = {
    "Bergruimte": "Overig",
    "Appartement": "Meergezinswoning",
    "Seniorenwoning": "Meergezinswoning",
    "Woonzorgwoning": "Meergezinswoning",
    "Serviceflatwoning": "Meergezinswoning",
    "Verzorgingscentra": "Meergezinswoning",
    "Begeleid wonen": "Meergezinswoning",
    "Meergezinshuis": "Meergezinswoning",
    "Maisonette": "Meergezinswoning",
    "Kamer": "Meergezinswoning",
    "Logeerkamer": "Meergezinswoningr",
    "Chalet": "Eengezinswoning",
    "Woonwagen": "Overig",
    "Standpl. woonwagen": "Overig",
    "Garage": "Overig",
    "Parkeerplaats": "Overig",
    "Overd. parkeerplaats": "Overig",
    "Bedrijfsruimte": "Overig",
    "Kantoor": "Overig",
    "Winkel": "Overig",
    "Praktijk": "Overig",
    "Peuterzaal": "Overig",
    "Kinderdagverblijf": "Overig",
    "Ontmoetingscentrum": "Overig",
    "Wijkgebouw": "Overig",
}

#Conditional imputation
df['Eengezins_Meergezins'] = df.apply(
    lambda row: impute_dict_eengezins_meergezins.get(row['Woning_type'], row['Eengezins_Meergezins']) 
    if pd.isna(row['Eengezins_Meergezins']) else row['Eengezins_Meergezins'],
    axis=1
)

#now eengezins_meergezins is around 0.42%

#Now i will look at VERA_type
#Using a dictionary again
impute_dict_VERA = {
    "Bergruimte": "Overig",
    "Appartement": "Woonruimte",
    "Seniorenwoning": "Woonruimte",
    "Woonzorgwoning": "Woonruimte",
    "Serviceflatwoning": "Woonruimte",
    "Verzorgingscentra": "Intramuraal zorgvastgoed",
    "Begeleid wonen": "Intramuraal zorgvastgoed",
    "Meergezinshuis": "Woonruimte",
    "Maisonette": "Woonruimte",
    "Kamer": "Woonruimte",
    "Logeerkamer": "Woonruimte",
    "Chalet": "Woonruimte",
    "Woonwagen": "Woonruimte",
    "Standpl. woonwagen": "Overig",
    "Garage": "Overig",
    "Parkeerplaats": "Overig",
    "Overd. parkeerplaats": "overig",
    "Bedrijfsruimte": "Bedrijfsruimte",
    "Kantoor": "Bedrijfsruimte",
    "Winkel": "Bedrijfsruimte",
    "Praktijk": "Bedrijfsruimte",
    "Peuterzaal": "Maatschappelijk vastgoed",
    "Kinderdagverblijf": "Maatschappelijk vastgoed",
    "Ontmoetingscentrum": "Maatschappelijk vastgoed",
    "Wijkgebouw": "Maatschappelijk vastgoed",
}

#Conditional imputation
df['VERA_Type'] = df.apply(
    lambda row: impute_dict_VERA.get(row['Woning_type'], row['VERA_Type']) 
    if pd.isna(row['VERA_Type']) else row['VERA_Type'],
    axis=1
)

#VERA went from 17% to 5%

#Doing some more conditional work based on Eengezins_Meergezins
df['VERA_Type'] = df.apply(
    lambda row: 'Woonruimte' if row['Eengezins_Meergezins'] in ['Eengezinswoning', 'Meergezinswoning'] and pd.isna(row['VERA_Type'])
    else row['VERA_Type'], axis=1
)
df['VERA_Type'] = df.apply(
    lambda row: 'Woonruimte' if pd.isna(row['VERA_Type']) and row['1e Slaapkamer'] > 0.0 else row['VERA_Type'], axis=1
)

#Now its from 5% to 0.22%
#Not adding binary flags for VERA_type, Eengezins_Meergezins and Woning_omschrijving because in principle, there is no situation in which these SHOULD
#Also didnt add placeholders because it doesnt make sense
#Doing KNN-imputation for the other 5% of missing Omschrijving_Vastgoed
#Again using source: https://www.geeksforgeeks.org/python-imputation-using-the-knnimputer/
# Label encoder

from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Step 1: Label encode 'Omschrijving_Vastgoed'
le = LabelEncoder()
df['Omschrijving_Vastgoed_encoded'] = le.fit_transform(df['Omschrijving_Vastgoed'].astype(str))

df.loc[df['Omschrijving_Vastgoed'].isna(), 'Omschrijving_Vastgoed_encoded'] = np.nan

#Features i want to use such as room sizes etc
features = df[['Omschrijving_Vastgoed_encoded', '1e Slaapkamer', '2e Slaapkamer', '3e Slaapkamer', 
               'Aparte douche/lavet+douche 1', 'Badkamer/doucheruimte 1', 'Toilet (Sanitair 1)', 
               'Totaal kamers', 'Totaal overige ruimtes', 'Verhuurbaar vloeroppervlakte', 'Woonkamer', 
               'Zolder', 'complexnummer']]

#Perform KNN imputation with k=5
imputer = KNNImputer(n_neighbors=5, weights="uniform") 
imputed_data = imputer.fit_transform(features)

#Impute back
df['Omschrijving_Vastgoed_encoded'] = np.where(df['Omschrijving_Vastgoed'].isna(), 
                                               imputed_data[:, 0],  
                                               df['Omschrijving_Vastgoed_encoded'])

#Decode
df['Omschrijving_Vastgoed'] = le.inverse_transform(df['Omschrijving_Vastgoed_encoded'].astype(int))

# Drop column
df.drop(columns=['Omschrijving_Vastgoed_encoded'], inplace=True)

#Categorical data is imputed
#Now to numeric data
#first lets clean the numeric values and remove all illegal values that i found in columns such as 0.0 for WOZ-values to see the true missing %
df['WOZ waarde'] = df['WOZ waarde'].replace([0.0], np.nan)
df['Markthuur'] = df['Markthuur'].replace([10.0], np.nan)
df['Maximaal_redelijke_huur'] = df['Maximaal_redelijke_huur'].replace([0.0,0.01], np.nan)
df['Streefhuur'] = df['Streefhuur'].replace([0.01,0.02,1.0,10.0,10.4], np.nan)


#Imputing max_redelijke_huur based on calculations provided by the government, for every point that the house has, you can sharge approximately 5.55 in rent
#Source=https://wetten.overheid.nl/BWBR0015386/2021-07-01#BijlageI

def impute_max_rent(df):
    df['Maximaal_redelijke_huur'] = df.apply(
        lambda row: row['Totale punten (onafgerond)'] * 5.55 
        if pd.isna(row['Maximaal_redelijke_huur']) else row['Maximaal_redelijke_huur'], 
        axis=1
    )

impute_max_rent(df)

#Streefhuur is around 70% of max_redelijke_huur, so max-redelijke huur is around 142% of streefhuur, if streefhuur is not missing and max redelijke huur is, im imputing it based on this information
#https://corporatiestrateeg.nl/corporatiebeleid/huurbeleid/wat-is-de-streefhuur/

df['Maximaal_redelijke_huur'] = np.where(
    df['Maximaal_redelijke_huur'].isna() & df['Streefhuur'].notna(),
    df['Streefhuur'] * 1.43, 
    df['Maximaal_redelijke_huur']  
)

df['Maximaal_redelijke_huur'] = df['Maximaal_redelijke_huur'].round(2)



#Missing % went from around 30/40% to 1.15%
#Now doing WOZ-values 

#Check gemeenten we have to map into regions: 

# Display all distinct items in the "Gemeente" column
#distinct_gemeente = df['Gemeente'].unique()

# Print each distinct item
#for gemeente in distinct_gemeente:
    #print(gemeente) #Needed this part for mapping gemeenten into regions but i dont want to run this every time, if you want to run it then remove hashtag 

#Now i will treat missing values for WOZ-waarde 
# Define the region for each municipality we have houses in

region_dict = {
    # Noord-Limburg
    'molenhoek': 'Noord-Limburg', 'mook': 'Noord-Limburg', 'middelaar': 'Noord-Limburg', 'plasmolen': 'Noord-Limburg',
    'gennep': 'Noord-Limburg', 'milsbeek': 'Noord-Limburg', 'heijen': 'Noord-Limburg', 'ottersum': 'Noord-Limburg', 
    'ven-zelderheide': 'Noord-Limburg', 'bergen': 'Noord-Limburg', 'well': 'Noord-Limburg', 'afferden': 'Noord-Limburg', 
    'siebengewald': 'Noord-Limburg', 'wellerlooi': 'Noord-Limburg', 'venray': 'Noord-Limburg', 'oostrum': 'Noord-Limburg', 
    'ysselsteyn': 'Noord-Limburg', 'leunen': 'Noord-Limburg', 'oirlo': 'Noord-Limburg', 'merselo': 'Noord-Limburg', 
    'castenray': 'Noord-Limburg', 'veulen': 'Noord-Limburg', 'heide': 'Noord-Limburg', 'vredepeel': 'Noord-Limburg', 
    'smakt': 'Noord-Limburg', 'wanssum': 'Noord-Limburg', 'blitterswijck': 'Noord-Limburg', 'geijsteren': 'Noord-Limburg',
    'meerlo': 'Noord-Limburg', 'tienray': 'Noord-Limburg', 'swolgen': 'Noord-Limburg', 'horst': 'Noord-Limburg', 
    'grubbenvorst': 'Noord-Limburg', 'melderslo': 'Noord-Limburg', 'america': 'Noord-Limburg', 'lottum': 'Noord-Limburg', 
    'hegelsom': 'Noord-Limburg', 'meterik': 'Noord-Limburg', 'broekhuizenvorst': 'Noord-Limburg', 
    'broekhuizen': 'Noord-Limburg', 'griendtsveen': 'Noord-Limburg', 'sevenum': 'Noord-Limburg', 
    'kronenberg': 'Noord-Limburg', 'evertsoord': 'Noord-Limburg', 'velden': 'Noord-Limburg', 'arcen': 'Noord-Limburg', 
    'lomm': 'Noord-Limburg', 'venlo': 'Noord-Limburg', 'tegelen': 'Noord-Limburg', 'belfeld': 'Noord-Limburg', 
    'steyl': 'Noord-Limburg', 'maasbree': 'Noord-Limburg', 'baarlo': 'Noord-Limburg','baarlo lb':'Noord-Limburg', 'panningen': 'Noord-Limburg', 
    'helden': 'Noord-Limburg', 'beringe': 'Noord-Limburg', 'grashoek': 'Noord-Limburg', 'egchel': 'Noord-Limburg', 
    'koningslust': 'Noord-Limburg', 'meijel': 'Noord-Limburg', 'kessel': 'Noord-Limburg', 
    'kessel-eik': 'Noord-Limburg', 'reuver': 'Noord-Limburg', 'offenbeek': 'Noord-Limburg', 'beesel': 'Noord-Limburg',

    # Midden-Limburg
    'nederweert': 'Midden-Limburg', 'ospel': 'Midden-Limburg', 'nederweert-eind': 'Midden-Limburg', 
    'leveroy': 'Midden-Limburg', 'ospeldijk': 'Midden-Limburg', 'weert': 'Midden-Limburg', 'stramproy': 'Midden-Limburg', 
    'altweerterheide': 'Midden-Limburg', 'tungelroy': 'Midden-Limburg', 'swartbroek': 'Midden-Limburg', 
    'laar': 'Midden-Limburg', 'ittervoort': 'Midden-Limburg', 'ell': 'Midden-Limburg', 'neeritter': 'Midden-Limburg', 
    'hunsel': 'Midden-Limburg', 'haler': 'Midden-Limburg', 'heythuysen': 'Midden-Limburg', 
    'baexem': 'Midden-Limburg', 'grathem': 'Midden-Limburg', 'kelpen-oler': 'Midden-Limburg', 
    'roggel': 'Midden-Limburg', 'neer': 'Midden-Limburg', 'heibloem': 'Midden-Limburg', 
    'haelen': 'Midden-Limburg', 'horn': 'Midden-Limburg', 'buggenum': 'Midden-Limburg', 
    'nunhem': 'Midden-Limburg', 'heel': 'Midden-Limburg', 'wessem': 'Midden-Limburg', 
    'beegden': 'Midden-Limburg', 'panheel': 'Midden-Limburg', 'thorn': 'Midden-Limburg', 
    'maasbracht': 'Midden-Limburg', 'linne': 'Midden-Limburg', 'stevensweert': 'Midden-Limburg', 
    'brachterbeek': 'Midden-Limburg', 'ohe en laak': 'Midden-Limburg', 'roermond': 'Midden-Limburg', 
    'maasniel': 'Midden-Limburg', 'herten': 'Midden-Limburg', 'merum': 'Midden-Limburg', 
    'leeuwen': 'Midden-Limburg', 'asenray': 'Midden-Limburg', 'ool': 'Midden-Limburg', 
    'swalmen': 'Midden-Limburg', 'boukoul': 'Midden-Limburg', 'asselt': 'Midden-Limburg', 
    'herkenbosch': 'Midden-Limburg', 'melick': 'Midden-Limburg', 'vlodrop': 'Midden-Limburg', 
    'montfort': 'Midden-Limburg', 'posterholt': 'Midden-Limburg', 'sint odilienberg': 'Midden-Limburg', 
    'echt': 'Midden-Limburg', 'susteren': 'Midden-Limburg', 'pey': 'Midden-Limburg', 
    'nieuwstadt': 'Midden-Limburg', 'koningsbosch': 'Midden-Limburg', 'sint joost': 'Midden-Limburg', 
    'roosteren': 'Midden-Limburg', 'maria hoop': 'Midden-Limburg', 'dieteren': 'Midden-Limburg',

    # Zuid-Limburg
    'sittard': 'Zuid-Limburg', 'geleen': 'Zuid-Limburg', 'born': 'Zuid-Limburg', 
    'munstergeleen': 'Zuid-Limburg', 'limbricht': 'Zuid-Limburg', 'grevenbicht': 'Zuid-Limburg', 
    'buchten': 'Zuid-Limburg', 'obbicht': 'Zuid-Limburg', 'einighausen': 'Zuid-Limburg', 
    'guttecoven': 'Zuid-Limburg', 'holtum': 'Zuid-Limburg', 'papenhoven': 'Zuid-Limburg', 
    'stein': 'Zuid-Limburg', 'elsloo': 'Zuid-Limburg', 'urmond': 'Zuid-Limburg', 
    'berg aan de maas': 'Zuid-Limburg', 'meers': 'Zuid-Limburg', 'beek': 'Zuid-Limburg', 'beek lb':'Zuid-Limburg',
    'spaubeek': 'Zuid-Limburg', 'neerbeek': 'Zuid-Limburg', 'genhout': 'Zuid-Limburg', 
    'geverik': 'Zuid-Limburg', 'oirsbeek': 'Zuid-Limburg', 'schinnen': 'Zuid-Limburg', 
    'amstenrade': 'Zuid-Limburg', 'puth': 'Zuid-Limburg', 'doenrade': 'Zuid-Limburg', 
    'sweikhuizen': 'Zuid-Limburg', 'nuth': 'Zuid-Limburg', 'hulsberg': 'Zuid-Limburg', 
    'schimmert': 'Zuid-Limburg', 'wijnandsrade': 'Zuid-Limburg', 'vaesrade': 'Zuid-Limburg', 
    'wijlre': 'Zuid-Limburg', 'pienk': 'Zuid-Limburg', 'kerkrade': 'Zuid-Limburg', 
    'landgraaf': 'Zuid-Limburg', 'hoensbroek': 'Zuid-Limburg', 'enckhuizen': 'Zuid-Limburg', 
    'heerlen': 'Zuid-Limburg', 'aubel': 'Zuid-Limburg', 'dalem': 'Zuid-Limburg', 
    'meerlo': 'Zuid-Limburg', 'nederweert': 'Zuid-Limburg', 'borchgrave': 'Zuid-Limburg', 
    'laak': 'Zuid-Limburg', 'haspel': 'Zuid-Limburg', 'guelders': 'Zuid-Limburg', 
    'gennep': 'Zuid-Limburg', 'landgraaf': 'Zuid-Limburg', 'bunde': 'Zuid-Limburg', 
    'hoofddorp': 'Zuid-Limburg', 'roermond': 'Zuid-Limburg', 'riemst': 'Zuid-Limburg', 
    'beekdaelen': 'Zuid-Limburg', 'hoensbroek': 'Zuid-Limburg','bocholtz':'Zuid-Limburg', 'brunssum':'Zuid-Limburg',
     'cadier en keer' : 'Zuid-Limburg', 'eygelshoven' : 'Zuid-Limburg',  'geulle' : 'Zuid-Limburg','jabeek':'Zuid-Limburg',
     'klimmen': 'Zuid-Limburg','merkelbeek': 'Zuid-Limburg','mheer': 'Zuid-Limburg','sint odiliÃ«nberg': 'Zuid-Limburg','voerendaal': 'Zuid-Limburg',
     'maastricht':'Zuid-Limburg',

    #Noord-Brabant

    'budel': 'Zuidoost-Noord-Brabant', 'helmond': 'Zuidoost-Noord-Brabant','maarheeze': ' Zuidoost-Noord-Brabant','someren': ' Zuidoost-Noord-Brabant','waalre': ' Zuidoost-Noord-Brabant',

    #gelderland

    'wijchen': 'Arnhem/Nijmegen'
}


# Normalize the 'Gemeente' values and impute
df['regio'] = df['Gemeente'].str.lower().str.strip().apply(lambda x: region_dict.get(x, 'Unknown Region'))


#Impute average WOZ-waarde per suqare meter based on averages per COROP gebied (Dictionary)
#At first missing % is 19%

impute_values = {
    'Noord-Limburg': 2164,
    'Midden-Limburg': 2121,
    'Zuid-Limburg': 2054,
    'Zuidoost-Noord-Brabant': 2866,
    'Arnhem/Nijmegen': 2893
}

# Impute missing values in 'WOZ waarde per m2'
df['WOZ waarde per m2'] = df['WOZ waarde per m2'].fillna(df['regio'].map(impute_values))


#write to cleaned data
df.to_csv('cleaned_data.csv', index=False)
cleaned_df =pd.read_csv('cleaned_data.csv')

#print to check the current missing values after treatment
print(display_missing_values(cleaned_df, max_columns=None, max_rows=None))

