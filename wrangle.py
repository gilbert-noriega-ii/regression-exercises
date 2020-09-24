import pandas as pd
import os
from env import host, user, password



def get_connection(db, user = user, host = host, password = password):
    '''
    This function connects to Sql and explores a database
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

###################### Acquire Telco Churn Data ########################

def new_telco_data():
    '''
    This function reads the telco churn data from CodeUp database into a df,
    write it to a csv file, and returns the df.
    '''
    sql_query = """
                SELECT customer_id, monthly_charges, tenure, total_charges
                FROM customers
                WHERE contract_type_id = 3
                """
    df = pd.read_sql(sql_query, get_connection('telco_churn'))
    df.to_csv('telco.csv')
    return df

def get_telco_data(cached=False):
    '''
    This function reads in telco churn data from CodeUp database if cached == False 
    or if cached == True reads in telco churn df from a csv file, returns df.
    '''
    if cached or os.path.isfile('telco.csv') == False:
        df = new_telco_data()
    else:
        df = pd.read_csv('telco.csv', index_col=0)
    return df

def wrangle_telco(cached=True):
    '''
    This function reads in telco churn data and preps it by 
    changing total_charges into float and handling null values
    '''
    # use my aquire function to read data into a df from a csv file
    df = get_telco_data(cached)
    # changing total_charges into a float
    df.total_charges = pd.to_numeric(df.total_charges, errors = 'coerce')
    # changing null values into 0's because tenure equals 0
    df.total_charges.fillna(0, inplace = True) 
    return df