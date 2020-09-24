from env import host, user, password
import seaborn as sns
import pandas as pd
import numpy as np
import os



def get_connection(db, user = user, host = host, password = password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

#################  Acquire Mall Customers Data  ##########################

def new_mall_data():
    '''
    This function reads the mall customers data from CodeUp database into a df,
    write it to a csv file, and returns the df.
    '''
    sql_query = 'SELECT * FROM customers'
    df = pd.read_sql(sql_query, get_connection('mall_customers'))
    df.to_csv('mall_customers_df.csv')
    return df

def get_mall_data(cached=False):
    '''
    This function reads in mall customer data from CodeUp database if cached == False 
    or if cached == True reads in mall customers df from a csv file, returns df.
    '''
    if cached or os.path.isfile('mall_customers_df.csv') == False:
        df = new_mall_data()
    else:
        df = pd.read_csv('mall_customers_df.csv', index_col=0)
    return df




#################  Acquire Titanic Data  ##########################
def new_titanic_data():
     '''
    This function reads the titanic data from CodeUp database into a df,
    write it to a csv file, and returns the df.
    '''
    sql_query = 'SELECT * FROM passengers'
    df = pd.read_sql(sql_query, get_connection('titanic_db'))
    df.to_csv('titanic.csv')
    return df

def get_titanic_data(cached=False):
    '''
    This function reads in titanic data from CodeUp database if cached == False 
    or if cached == True reads in mall customers df from a csv file, returns df.
    '''
    if cached or os.path.isfile('titanic.csv') == False:
        df = new_titanic_data()
    else:
        df = pd.read_csv('titanic.csv', index_col=0)
    return df

#################  Acquire Iris Data  ##########################


def new_iris_data():
     '''
    This function reads the iris data from CodeUp database into a df,
    write it to a csv file, and returns the df.
    '''
    sql_query = 'SELECT * FROM measurements AS m JOIN species USING (species_id)'
    df = pd.read_sql(sql_query, get_connection('iris_db'))
    df.to_csv('iris.csv')
    return df

def get_iris_data(cached=False):
    '''
    This function reads in iris data from CodeUp database if cached == False 
    or if cached == True reads in mall customers df from a csv file, returns df.
    '''
    if cached or os.path.isfile('iris.csv') == False:
        df = new_iris_data()
    else:
        df = pd.read_csv('iris.csv', index_col=0)
    return df
