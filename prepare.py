import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from acquire import get_titanic_data, get_iris_data

from wrangle import wrangle_telco
import sklearn.preprocessing


##################### Prep Wrangle Telco Data ##################

def telco_split(df):
    train_and_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_and_validate, test_size=.25, random_state=123)
    return train, validate, test

def scale_wrangle_telco(cached=True):
    '''
    This function acquires wrangle_telco data, 
    splits into train, validate, and test,
    scales the numeric columns using min-max scaling,
    and adds the scaled columns to the respective split data sets
    '''
    #acquires 'wrangle_telco' and saves it as df
    df = wrangle_telco(cached)
    #uses the function above to split the into train, validate and test
    train, validate, test = telco_split(df)
    #assigns the scaling method as min-max scaler
    scaler = sklearn.preprocessing.MinMaxScaler()
    #identifies the columns to scale
    columns_to_scale = ['monthly_charges', 'tenure', 'total_charges']
    #adds '_scaled' to the end of the newly scaled columns to identify differences
    new_column_names = [c + '_scaled' for c in columns_to_scale]
    #fts the columns to the scaler
    scaler.fit(train[columns_to_scale])
    #concatonates the newly created scaled columns to their respective data sets,
    #adds 'new_column_names' as the label to the added columns
    #uses the original index since the new columns no longer have an index
    train = pd.concat([
        train,
        pd.DataFrame(scaler.transform(train[columns_to_scale]), columns=new_column_names, index=train.index),
    ], axis=1)
    validate = pd.concat([
        validate,
        pd.DataFrame(scaler.transform(validate[columns_to_scale]), columns=new_column_names, index=validate.index),
    ], axis=1)
    test = pd.concat([
        test,
        pd.DataFrame(scaler.transform(test[columns_to_scale]), columns=new_column_names, index=test.index),
    ], axis=1)
    #returns the data sets with the new respective scaled data
    return train, validate, test



##################### Prep Mall Customer Data ##################

def prep_mall_data(df):
    '''
    Takes the acquired mall data, does data prep, and returns
    train, test, and validate data splits
    '''
    df['is_female'] = (df.gender == 'Female').astype('int')
    train_and_validate, test = train_test_split(df, test_size = .15, random_state = 123)
    train, validate = train_test_split(train_and_validate, test_size = .15, random_state = 123)
    return train, test, validate

###################### Prep Iris Data ######################

def prep_iris_data(cached=True):
    
    # use my aquire function to read data into a df from a csv file
    df = get_iris_data(cached)
    
    # drop and rename columns
    df = df.drop(columns='species_id').rename(columns={'species_name': 'species'})
    
    # create dummy columns for species
    species_dummies = pd.get_dummies(df.species, drop_first=True)
    
    # add dummy columns to df
    df = pd.concat([df, species_dummies], axis=1)
    
    return df

###################### Prep Titanic Data ######################

def titanic_split(df):

    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123, 
                                        stratify=df.survived)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123, 
                                   stratify=train_validate.survived)
    return train, validate, test



def impute_mean_age(train, validate, test):

    # create the imputer object with mean strategy
    imputer = SimpleImputer(strategy = 'mean')
    
    # fit on and transform age column in train
    train['age'] = imputer.fit_transform(train[['age']])
    
    # transform age column in validate
    validate['age'] = imputer.transform(validate[['age']])
    
    # transform age column in test
    test['age'] = imputer.transform(test[['age']])
    
    return train, validate, test



def prep_titanic_data(cached=True):
    '''
    Takes the titanic data, does data prep, and returns
    train, test, and validate data splits
    '''

    # use my acquire function to read data into a df from a csv file
    df = get_titanic_data(cached)
    
    # drop rows where embarked/embark town are null values
    df = df[~df.embarked.isnull()]
    
    # encode embarked and sex using dummy columns
    titanic_dummies = pd.get_dummies(df[['sex', 'embarked']], drop_first=True)
    
    # join dummy columns back to df
    df = pd.concat([df, titanic_dummies], axis=1)
    
    # drop the deck column
    df = df.drop(columns=['deck', 'sex', 'embarked', 'class', 'embark_town', 'passenger_id'])
    
    # # split data into train, validate, test dfs
    train, validate, test = titanic_split(df)
    
    # # impute mean of age into null values in age column
    train, validate, test = impute_mean_age(train, validate, test)
    
    return train, validate, test


    #################### Scale Any Data Set ##################
    def add_scaled_columns(train, validate, test, scaler, columns_to_scale):
        new_column_names = [c + '_scaled' for c in columns_to_scale]
        scaler.fit(train[columns_to_scale])

        train = pd.concat([
            train,
            pd.DataFrame(scaler.transform(train[columns_to_scale]), columns=new_column_names, index=train.index),
        ], axis=1)
        validate = pd.concat([
            validate,
            pd.DataFrame(scaler.transform(validate[columns_to_scale]), columns=new_column_names, index=validate.index),
        ], axis=1)
        test = pd.concat([
            test,
            pd.DataFrame(scaler.transform(test[columns_to_scale]), columns=new_column_names, index=test.index),
        ], axis=1)
    
        return train, validate, test