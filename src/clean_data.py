import pandas as pd
import numpy as np

import argparse
import re
import math

from sklearn  import preprocessing, impute
from datetime import datetime

### Constants ###

DATA_DIR = "../data/kaggle"


### Functions ###

def get_outfile_name( in_file ):

    f_split = in_file.split(".")
    now = datetime.now()

    s_date = now.strftime("%Y%m%d")
    s_time = now.strftime("%H%M%S")

    return f"{f_split[0]}.clean.{s_date}.{s_time}.{f_split[1]}"

def split_last_name( row ):
    l_subs = row['Name'].split(r', ')
    return l_subs

def split_title( row ):
    l_subs = row[1].split(r'. ')
    if len(l_subs) > 2:
        l_subs[1] = f"{l_subs[1]} {l_subs[2]}"
        l_subs.pop()
    return l_subs

def split_cabin_deck( row ):

    if type(row) is float:
        return ""
    else:
        # split the letters into a list
        l_letters = re.findall( "[a-zA-Z]+", row )

        # check if all of the letters are the same; in rare cases where they
        # are not the same, the first character is "F", so we're choosing the
        # second
        if all(x == l_letters[0] for x in l_letters):
            return l_letters[0]
        else:
            return l_letters[1]

def split_cabin_room( row ):

    if type(row) is float:
        return ""
    else:
        # split the letters into a list
        l_numbers = re.findall( "[0-9]+", row )

        if len(l_numbers) == 0:
            return ""
        else:
            median = math.ceil((len(l_numbers) - 1) / 2)
            if all(x == l_numbers[0] for x in l_numbers):
                return int(l_numbers[0])
            else:
                return int(l_numbers[median])

def print_subset( df ):

    #df.sort_values(['GroupSize', 'LastName'], ascending=False, inplace=True)
    print(df.columns)
    print(df[ ["LastName", "Survived", "TicketSurvivorScore", "Title",
               "Pclass", "FamilySize", "TicketFrequency", "Ticket" ]].to_string())

    return True

def get_p3_or_dead_title( row ):
    l_dead_titles = [ 'Capt', 'Rev', 'Mr' ]

    s_title = row['Title']
    b_pclass3 = row['Pclass3']

    if ( s_title in l_dead_titles ):
        if ( b_pclass3 == 1 ):
            return 0
        else:
            return 1
    else:
        return 2

def get_other_titles( row ):

    l_titles = [ 'Mr', 'Mrs', 'Miss', 'Master' ]
    s_title = row['TitleGrouped']

    if (not s_title in l_titles):
        if row['Sex'] == 'male':
            return 'OtherMale'
        else:
            return 'OtherFemale'
    else:
        return s_title

def get_value_frequency_df(df, col_name):

    col_freq = df[[col_name]].value_counts()
    df_freq = col_freq.reset_index()

    freq_col_name = col_name + "Frequency"
    df_freq.columns = [col_name, freq_col_name]

    return df_freq

def get_groupby_sum_df( df, x_col_name, y_col_name ):

    df = df[ df["TicketFrequency"] > 1 ]

    xcol_sum = df.groupby([x_col_name])[y_col_name].sum()
    df_sum = xcol_sum.reset_index()

    sum_col_name = x_col_name + "Confirmed" + y_col_name
    df_sum.columns = [x_col_name, sum_col_name]

    return df_sum

def get_column_dummies( df, s_column ):

    df_temp = pd.DataFrame(df[["PassengerId", s_column ]])

    df_titles = pd.get_dummies(df_temp[s_column])
    df_titles["PassengerId"] = df_temp['PassengerId']
    df_titles = df_titles.map(lambda x: int(x))

    return df_titles

def scaler_fit_transform( scaler, df ):

    l_transformed = scaler.fit_transform(df)
    return l_transformed[:,0].round(4)


### Main ###

parser = argparse.ArgumentParser( description='Pre-process data from Titanic dataset' )
parser.add_argument('-w', '--write', action='store_true', help='Write processed output to files')

args = vars( parser.parse_args() )

# load train data
df_train = pd.read_csv(f"../data/kaggle/train.csv")

# load test data with a Survived column to encode data
df_test  = pd.read_csv(f"../data/kaggle/test.csv")
df_test["Survived"] = np.nan

df_full = pd.concat([df_train, df_test])

# preprocess categorical data (sex and title) into ordinal values
title_enc = preprocessing.OrdinalEncoder(
    categories=[['Mr', 'OtherMale', 'Master', 'Miss', 'Mrs', 'OtherFemale']])
enc = preprocessing.OrdinalEncoder()
#df_full = get_name_and_title( df_full )

df_temp_name = df_full.apply( split_last_name, axis=1, result_type='expand' )
df_temp_title = df_temp_name.apply( split_title, axis=1, result_type='expand' )

df_full["LastName"] = df_temp_name[0]
df_full["Title"] = df_temp_title[0]

df_full['TitleGrouped'] = df_full['Title']
df_full['TitleGrouped'] = df_full.apply( get_other_titles, axis=1 )

df_full["LastNameOrd"] = enc.fit_transform(df_full[["LastName"]])
df_full["TitleOrd"] = title_enc.fit_transform(df_full[["TitleGrouped"]])
df_full["SexOrd"] = enc.fit_transform(df_full[["Sex"]])

# create dummy variables / categories for the 5 title groupings
df_titles = get_column_dummies( df_full, 'TitleGrouped' )
df_full = pd.merge(df_full, df_titles, on="PassengerId")

df_full["Died"] = df_full["Survived"].apply( lambda v: 1 if v == 0 else 0 )

x_colnames = [ "Ticket", "LastName" ]

for c in x_colnames:

    df_frequency = get_value_frequency_df( df_full, c)

    df_full = pd.merge(df_full, df_frequency, how='left', on=c)

    df_survived = get_groupby_sum_df( df_full, c, "Survived" )
    df_full = pd.merge(df_full, df_survived, how='left', on=c)

    df_died = get_groupby_sum_df( df_full, c, "Died" )
    df_full = pd.merge(df_full, df_died, how='left', on=c)

    df_full[c+"SurvivorScore"] = df_full[c+"ConfirmedSurvived"] - df_full[c+"ConfirmedDied"]
    df_full[c+"SurvivorScore"] = df_full[c+"SurvivorScore"].fillna(value=0)

df_full["IsMale"]        = df_full["Sex"].apply( lambda v: 1 if v == "male" else 0)
df_full["IsChild"]       = df_full["Age"].apply( lambda v: 1 if v <= 16 else 0)
df_full["IsYoungChild"]  = df_full["Age"].apply( lambda v: 1 if v <= 8 else 0)
df_full["FamilySize"]     = df_full["SibSp"] + df_full["Parch"] + 1
df_full["LargeGroup"]    = df_full["TicketFrequency"].apply( lambda v: 1 if v >= 5 else 0)
df_full["SmallGroup"]    = df_full["TicketFrequency"].apply( lambda v: 1 if (v > 1 and v < 5) else 0 )
df_full["IsSolo"]        = df_full["TicketFrequency"].apply( lambda v: 1 if v == 1 else 0 )
df_full["NoFamily"]      = df_full["FamilySize"].apply( lambda v: 0 if v > 1 else 1 )
df_full["Fare"]          = df_full["Fare"].apply( lambda v: 0 if np.isnan(v) else v )
df_full["FarePerPerson"] = round((df_full["Fare"] / df_full["TicketFrequency"]),4)

df_full["Pclass1"]       = df_full["Pclass"].apply( lambda v: 1 if v == 1 else 0 )
df_full["Pclass2"]       = df_full["Pclass"].apply( lambda v: 1 if v == 2 else 0 )
df_full["Pclass3"]       = df_full["Pclass"].apply( lambda v: 1 if v == 3 else 0 )

df_full["P3orDeadTitle"] = df_full.apply( get_p3_or_dead_title, axis=1 )

df_full["HasCabin"]      = df_full["Cabin"].apply( lambda v: 0 if pd.isna(v) else 1 )
df_full["CabinDeck"]     = df_full["Cabin"].apply( split_cabin_deck )
df_full["CabinRoom"]     = df_full["Cabin"].apply( split_cabin_room )

df_full["CabinDeck"] = df_full["CabinDeck"].fillna(value="X")
df_full["CabinOrd"] = enc.fit_transform(df_full[["CabinDeck"]])

# create dummy variables / categories for the 5 title groupings
df_cabins = get_column_dummies( df_full, 'CabinDeck' )
for i in ["A", "B", "C", "D", "E", "F", "G", "T"]:
    df_full["Cabin" + i ] = df_cabins[i]
#df_full = pd.merge(df_full, df_cabins, on="PassengerId")

# create dummy variables / categories for the 5 title groupings
df_full["Embarked"] = df_full["Embarked"].fillna(value="S")
df_embarked = get_column_dummies( df_full, 'Embarked' )
for i in ["C", "Q", "S" ]:
    df_full["Embark" + i ] = df_embarked[i]
df_full["EmbarkOrd"] = enc.fit_transform(df_full[["Embarked"]])

imputer = impute.KNNImputer(n_neighbors=5)
l_imputed = imputer.fit_transform(df_full[[ "Age", "Mr", "Mrs", "Miss", "Master", "OtherMale", "OtherFemale", "Pclass" ]])

df_full["AgeImputed"] = l_imputed[:,0].round(4)

# normalize the age and fare per person features, to reduce impact of outliers
robust = preprocessing.RobustScaler()
minmax = preprocessing.MinMaxScaler()

df_full["AgeRobust"] = scaler_fit_transform( robust, df_full[["AgeImputed"]] )
df_full["AgeMinMax"] = scaler_fit_transform( minmax, df_full[["AgeImputed"]] )

df_full["FppRobust"] = scaler_fit_transform( robust, df_full[["FarePerPerson"]] )
df_full["FppMinMax"] = scaler_fit_transform( minmax, df_full[["FarePerPerson"]] )

#print_subset(df_full[ df_full["TicketFrequency"] > 5].head(40))

# prep the cleaned training data to write
df_train_clean = df_full[ df_full[ "Survived" ].notna() ].copy()

# prep the cleaned test data to write
df_test_clean = df_full[ df_full[ "Survived" ].isna() ].copy()
df_test_clean.drop( "Survived", axis=1, inplace=True )

if args["write"]:

    train_outfile = get_outfile_name("train.csv")
    print(f"write train shape: {df_train_clean.shape} to file: {train_outfile}")
    df_train_clean.to_csv(f"../data/kaggle/{train_outfile}", index=False)

    test_outfile = get_outfile_name("test.csv")
    print(f"write test shape:  {df_test_clean.shape} to file: {test_outfile}")
    df_test_clean.to_csv(f"../data/kaggle/{test_outfile}", index=False)

else:

    print(df_train_clean.info())

    print("No output written to files")
    print(f"  train shape: {df_train_clean.shape}")
    print(f"  test shape:  {df_test_clean.shape}")
