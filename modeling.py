import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler


########################### SelectKBest Functions ###########################

def select_kbest(predictors, target, number_of_features):
    '''
    This function takes in predictors(features), a target variable and the number of top features we want
    and returns the top features that correlate with the target variable.
    '''
    #Initialize the f_selector object, 
    #which defines the test for scoring the features 
    #and the number of features we want to keep,
    f_selector = SelectKBest(f_regression, k = number_of_features)
    
    #fitting the data into the model
    #scoring, ranking and identifying the top k features
    f_selector = f_selector.fit(predictors, target)
    
    #creating a list of the features that remain
    f_support = f_selector.get_support()

    #We get a list of the feature names selected from 
    #X_train using .loc with our mask, 
    #using .columns to get the column names, 
    #and convert the values to a list using .tolist()
    f_feature = predictors.iloc[:, f_support].columns.tolist()

    return f_feature


########################### RFE Functions ###########################

def rfe(predictors, target, number_of_features):
    '''
    This function takes in predictors(features), a target variable and the number of top features we want 
    and returns the top features that lead to the best performing linear regression model. 
    '''
    #Initialize the linear regression object
    lm = LinearRegression()
    
    #Initialize the RFE object, 
    #setting the hyperparameters to be our linear regression 
    #(as the algorithm to test the features on) 
    #and the number of features to be returned
    rfe = RFE(lm, number_of_features)

    #Fit the RFE object to our data. 
    #(This means create multiple linear regression models,
    #find the one that performs best, 
    #and identify the predictors that are used in that model.
    #Those are the features we want.)
    #Transform our X dataframe to include only 
    #the 'number_of_features' that performed the best
    X_rfe = rfe.fit_transform(predictors, target)

    #Create a mask to hold a list of the features that remain
    mask = rfe.support_

    #We get a list of the feature names selected from 
    #X_train using .loc with our mask, 
    #using .columns to get the column names, 
    #and convert the values to a list using .tolist()
    X_reduced_scaled_rfe = predictors.iloc[:, mask].columns.tolist()

    return X_reduced_scaled_rfe


########################### Scaling Functions ###########################

def scaling(train, validate, test):
    '''
    This function takes in a train, validate and test dataframe and returns 
    the scaled versions of each dataframe respectfully
    '''
    #create the scaler
    scaler = MinMaxScaler(copy=True)
    
    
    #fitting the train dataframe
    scaler.fit(train)
    #scaling the data
    train_scaled = scaler.transform(train)
    #storing the scaled data as a dataframe
    train_scaled = pd.DataFrame(train_scaled, 
             columns = train.columns.values).\
             set_index(train.index.values)
    
    
    #fitting the validate dataframe
    scaler.fit(validate)
    #scaling the data
    validate_scaled = scaler.transform(validate)
    #storing the scaled data as a dataframe
    validate_scaled = pd.DataFrame(validate_scaled, 
             columns = validate.columns.values).\
             set_index(validate.index.values)
    
    #fitting the test dataframe
    scaler.fit(test)
    #scaling the data
    test_scaled = scaler.transform(test)
    #storing the scaled data as a dataframe
    test_scaled = pd.DataFrame(test_scaled, 
             columns = test.columns.values).\
             set_index(test.index.values)
    return train_scaled, validate_scaled, test_scaled