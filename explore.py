import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math


######################### Explore Functions #########################

def plot_variable_pairs(df, drop_scaled_columns = True):
    '''
    This function takes in a DataFrame and plots all of the 
    pairwise relationships along with the regression line for each pair.
    '''
    if drop_scaled_columns:
        scaled_columns = [c for c in df.columns if c.endswith('_scaled')]
        df = df.drop(columns = scaled_columns)
    #to see all the plots at once, pairplot but with more customizations
    g = sns.PairGrid(df)
    #the plots is the diagonal will be a distribution plot
    g.map_diag(plt.hist) #one for a single variable
    #the plots not in the diagonal will be a scatter plot
    g.map_offdiag(sns.regplot) #one for the interaction of two variables
    plt.show()
    return g


def month_to_years(df):
    '''
    This function accepts a dataframe and returns a dataframe 
    with a new feature tenure_years, in complete years as a customer.
    '''
    # math floor rounds down to the nearest whole number
    # astype changes the number to an integer
    df['tenure_years'] = (df.tenure / 12).apply(math.floor)
    return df

def plot_categorical_and_continuous_vars(df, continous, categorical):
    '''
    This function accepts a dataframe and the name of the columns 
    that hold the continuous and categorical features 
    and outputs 4 different plots for visualizing a 
    categorical variable and a continuous variable.
    '''
    sns.boxplot(data = df, y = continous, x = categorical)
    plt.show()
    sns.barplot(data = df, y = continous, x = categorical)
    plt.show()
    sns.swarmplot(data = df, y = continous, x = categorical)
    plt.show()
    sns.violinplot(data = df, y = continous, x = categorical)
    plt.show()