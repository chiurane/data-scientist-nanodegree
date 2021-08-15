import sys
import os
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """ 
    Loads and merges data from the two files 
    params
    ---------------------------------------------------------
    messages_filepath:  file path to messages csv file
    categories_filepath: file path to categories csv file
    
    ouput
    df: merged messages and categories
    """
    
    # load the messages and categories
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge messages and categories
    df = pd.merge(messages, categories, on='id')
    
    return df


def clean_data(df):
    """
    Performs clean-up operations on the dataframe
    params:
    ----------------------------------------------------------
    df: the dataframe to clean
    output
    df: cleaned dataframe
    """
    
    # split categories into separate category columns
    categories = df['categories'].str.split(';', expand = True)
    
    # select the first row of the categories dataframe
    row = categories.loc[0,:]
    
    # use this row tp extract a list of new column names for categories
    # here we apply a lambda function that slices the data and takes first part
    category_colnames = list(row.apply(lambda x: x.split('-')[0]))
    
    # rename the columns of categories series
    categories.columns = category_colnames
    
    # convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x.split('-')[1])
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # replace categories column in df with new category columns
    df.drop(columns = ['categories'], inplace = True)
    
    # concatenate the original dataframe with the new categories dataframe
    df = pd.concat([df, categories], axis = 1)
    
    # remove duplicates
    df = df.drop_duplicates(keep = 'first')
    
    df['related'] = df['related'].replace(2,1)
    
    return df # return the cleaned dataframe


def save_data(df, database_filename):
    """
    Saves our clean to a sqlite database
    params:
    -------------------------------------------------------------------
    df:     dataframe to save to the database
    database_filename: sqlite database filename
    """
    
    # check if the database already exists and delete
    if os.path.exists(database_filename):
        os.remove(database_filename)
        
    # instance of engine to database
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('message', engine, index = False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        messages_filepath = os.path.join(os.getcwd(), os.path.join('data', messages_filepath))
        categories_filepath = os.path.join(os.getcwd(), os.path.join('data', categories_filepath))
        database_filepath = os.path.join(os.getcwd(), os.path.join('data', database_filepath))

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()