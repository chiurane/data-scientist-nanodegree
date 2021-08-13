import sys
import os
import nltk
import sqlite3
import pickle
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

from app.utils import tokenize

#nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

def load_data(database_filepath):
    """
    Load data from the sqlite database
    params:
    ---------------------------------------------------------
    INPUT
    database_filepath: path to the sqlite3 database
    
    OUTPUT
    X, Y: dataframes with X as the feature and Y as the target labels
    category_names: label names for the target
    """
    
    # Update path to parent directory
    #database_filepath = os.path.join(os.pardir, database_filepath)
    
    # instance to the database engine
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    
    # read form the database table
    df = pd.read_sql_table('message', con = engine)
    
    # create the X and Y dataframes
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    
    # extract feature names from Y
    category_names = Y.columns
    
    # return X, Y, category_names
    return X, Y, category_names
    
def build_model():
    """
    build a machine learning pipeline here
    params:
    -------------------------------------------------------------
    OUTPUT
    cv: GridSearchCV optimized pipeline
    """
    
    # create the pipeline that users MultiOutputClassifier and KNeighborsClassifier as 
    # classifier
    pipeline = Pipeline(
        [('vect', CountVectorizer(tokenizer = tokenize)),
         ('tfidf', TfidfTransformer()),
         ('clf', MultiOutputClassifier(KNeighborsClassifier()))
        ])
    
    # GridSearchCV optimization parameters
    parameters = {'tfidf__use_idf': (True, False),
                  'vect__max_features':(None, 5000, 10000)}
    
    # Our GridSearch parameter search here
    cv = GridSearchCV(pipeline, param_grid = parameters, verbose = 10)
    
    # return CV optimized estimator
    return cv
    
def evaluate_model(model, X_test, Y_test, category_names):
    """ Evaluate our model 
    params
    -------------------------------------------------------------------
    INPUT
    model: fitted model
    X_test: X test set
    Y_test: Y test set
    category_names: label names
    OUTPUT
    
    """
    
    # lets get predictions from our model
    Y_pred = model.predict(X_test)
    
    # print out the results of the model evaluation
    print(classification_report(Y_test, Y_pred, target_names = category_names))

def save_model(model, model_filepath):
    """
    save our model to a pickle file
    params
    ------------------------------------------------------------------
    INPUT
    model: trained model
    model_filepath: file path to save model to
    OUTPUT
    
    """
    # check if the pickle file already exists and delete it, if it does
    if os.path.exists(model_filepath):
        # delete it
        os.remove(model_filepath)
    
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        # path to files
        database_filepath, model_filepath = sys.argv[1:]
        database_filepath = os.path.join(os.getcwd(), os.path.join('data', database_filepath))
        model_filepath = os.path.join(os.getcwd(), os.path.join('models', model_filepath))
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))

        # load the data
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
