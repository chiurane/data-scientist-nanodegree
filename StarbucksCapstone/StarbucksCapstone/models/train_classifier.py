import sqlalchemy
import pickle
import sys
import os
from sqlalchemy import create_engine
import pandas as pd
import numpy as np

def load_data(database_filepath, table_name):
    """
    Load data from the Starbucks database
    params:
    ---------------------------------------------------------
    INPUT
    database_filepath: path to the sqlite3 database

    OUTPUT
    df - our cleaned and engineered dataframe from process_data
    """
    # instance to the database engine
    engine = create_engine('sqlite:///{}'.format(database_filepath))

    # read form the database table
    df = pd.read_sql_table(table_name, con=engine)

    return df # return our df

def save_model(model, model_filepath):
    """
    saves models to a pickle file
    params
    ------------------------------------------------------------------
    INPUT
    model: a model to save
    model_filepath: file path to save model to
    OUTPUT

    """
    # check if the pickle file already exists and delete it, if it does
    if os.path.exists(model_filepath):
        # delete it
        os.remove(model_filepath)

    # Lets write to pickle file
    with open(model_filepath, 'wb') as obj:
        pickle.dump(model, obj)

def create_user_item_matrix(df_transcript, df_portfolio, out_file='user_item_matrix.p'):
    """
    Create the user-item matrix with 1's and 0's and writes it to file
    INPUT:
    df - pandas dataframe with our cleaned transcript

    OUTPUT:
    user_item - user item matrix

    Description:
    Return a matrix with person_ids as rows and offers as columns with a
    1 where user interacted with an offer 3 times (offer received, offer viewed and offer completed)
    and 0 otherwise
    """

    # Base user-item matrix
    user_item_matrix = df_transcript.groupby(['person', 'offer_id'])['event'].count().unstack()

    # Fill all nulls with zeros
    user_item_matrix = user_item_matrix.fillna(0)

    # Lets ignore informational offers in the portfolio for now
    user_item_matrix.drop(list(df_portfolio[df_portfolio['offer_type'] == 'informational']['id']), axis=1, inplace=True)

    # We go through all our data to find the desired offered received, offer viewed
    # offer completed combinations
    for offer_id in user_item_matrix.columns.values:
        for person_id in user_item_matrix.index:

            if user_item_matrix.loc[person_id, offer_id] >= 3:
                # We just look for users who executed our desired workflow in the
                # order offer recieved, offer viewed and offer completed
                events = [event for event in
                          df_transcript[(df_transcript['offer_id'] == offer_id) & (df_transcript['person'] == person_id)][
                              'event']]
                user_item_matrix.loc[person_id, offer_id] = 0

                # We just look for a tripple here
                size = len(events)
                for i in range(size - 2):
                    if (events[i] == 'offer received') & (events[i + 1] == 'offer viewed') & (
                            events[i + 2] == 'offer completed'):
                        user_item_matrix.loc[person_id, offer_id] += 1
            else:
                user_item_matrix.loc[person_id, offer_id] = 0

    # Convert all the number
    for col in user_item_matrix.columns.values:
        user_item_matrix[col] = (user_item_matrix[col] >= 1).astype(int).astype(float)

    # Lets write our matrix to file to save time for later
    save_model(user_item_matrix, out_file)

    return user_item_matrix  # return the user_item_matrix


def train_test_split(df, test_size=0.3):
    """
    Lets split our dataframe into a train and test set
    INPUT:
    df - pandas dataframe to split
    OUTPUT:
    train - train pandas dataframe
    test - test pandas data frame
    train, test - the train and test dataframes
    Description:
    Splits a dataframe into a train and test set using the test size to
    determine our split fraction
    """
    # split df here
    train_size = int(df.shape[0] * (1 - test_size))
    test_size = df.shape[0] - train_size
    train = df[:train_size]
    test = df[train_size:]

    return train, test  # return the train and test datasets


def create_train_and_test_user_item_matrix(df_transcript, df_portfolio, test_size=.3):
    """
    INPUT:
    df - pandas dataframe
    test_size - test size to use for splitting
    OUTPUT:
    user_item_train - a user-item matrix for the training dataframe
    user_item_test - a user-item matrix for the test dataframe
    test_idx - all of the test user ids
    test_offers - all the test offer ids
    """

    train, test = train_test_split(df_transcript, test_size=test_size)

    # user_item_train and user_item_test
    user_item_train = create_user_item_matrix(train, df_portfolio, 'user_item_train.p')
    user_item_test = create_user_item_matrix(test, df_portfolio, 'user_item_test.p')

    # test_idx
    test_idx = user_item_test.index
    test_offers = user_item_test.columns

    return user_item_train, user_item_test, test_idx, test_offers


def FunkSVD(script_mat, latent_features=18, learning_rate=0.0001, iters=100):
    """
    INPUT:
    script_mat - a numpy array (matrix) with users as rows, offers as columns and values being wether or not they interacted
    latent_features - (int) the number of latent features used
    learning_rate = (float) the learning rate
    iters - (int) the number of iterations

    OUTPUT:
    user_mat - (numpy array) a user by latent features matrix
    portfolio_mat - (numpy array) a latent feature by product marix

    Description:
    This function performs matrix factorization using a basic form of FunkSVD with no regularization
    """
    # Set up useful values to be used through the rest of the function
    n_users = script_mat.shape[0]  # number of rows in the matrix
    n_portfolio = script_mat.shape[1]  # number of movies in the matrix
    num_interactions = np.count_nonzero(~np.isnan(script_mat))  # total number of ratings in the matrix

    # initialize the user and movie matrices with random values
    # helpful link: https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.rand.html
    user_mat = np.random.rand(n_users, latent_features)  # user matrix filled with random values of shape user x latent
    portfolio_mat = np.random.rand(latent_features,
                                   n_portfolio)  # movie matrix filled with random values of shape latent x movies

    # initialize sse at 0 for first iteration
    sse_accum = 0
    mse = 0

    # header for running results
    #print("Optimization Statistics")
    #print("Iterations | Mean Squared Error ")

    # for each iteration
    for iteration in range(iters):

        # update our sse
        old_sse = sse_accum
        sse_accum = 0

        # For each user-movie pair
        for i in range(n_users):
            for j in range(n_portfolio):

                # if the rating exists
                if script_mat[i, j] > 0:

                    # compute the error as the actual minus the dot product of the user and movie latent features
                    diff = script_mat[i, j] - np.dot(user_mat[i, :], portfolio_mat[:, j])

                    # Keep track of the total sum of squared errors for the matrix
                    sse_accum += diff ** 2

                    # update the values in each matrix in the direction of the gradient
                    for k in range(latent_features):
                        user_mat[i, k] += learning_rate * (2 * diff * portfolio_mat[k, j])
                        portfolio_mat[k, j] += learning_rate * (2 * diff * user_mat[i, k])

        # print results for iteration
        mse = sse_accum / num_interactions
        #print("%d \t\t %f" % (iteration+1, sse_accum / num_interactions))

    return (user_mat, portfolio_mat, mse)

def fit(df_transcript, df_portfolio):
    """"
    INPUT:
    df_transcript: the transcript dataframe
    df_portfolio: the portfolio dataframe
    OUTPUT
    train_df - the original user_item_train to be used for making predictions on the training dataset
    test_df: the original user_item_test dataset for testing
    Description:
    Builds our FunkSVD model
    """
    # first we get the user_item matrices
    user_item_train, user_item_test, test_idx, test_offers = create_train_and_test_user_item_matrix(df_transcript,
                                                                                                    df_portfolio)

    # Convert our user_item_train dataframe to numpy array
    train_df = user_item_train.copy()
    test_df = user_item_test.copy()

    # Convert our user_item_train dataframe to numpy array
    user_item_train = np.array(user_item_train)

    # Fit using FunkSVD
    user_mat, portfolio_mat, mse = FunkSVD(user_item_train, latent_features=27, learning_rate=0.0001, iters=250)

    # return all our results here
    return train_df, test_df, user_item_train, user_item_test, test_idx, test_offers, user_mat, portfolio_mat, mse


def predict(df, user_mat, portfolio_mat, user_id, offer_id):
    """
    INPUT:
    df - the source dataset for the test (can be train or test)
    user_mat - user by latent factor matrix from FunkSVD
    protfolio_mat - latent factor by portfolio matrix
    user_id - the user to make prediction for
    offer_id - the offer id
    OUTPUT:
    pred - the prediction for this user and offer based on our model
    """
    try:
        # Lets create the user and portfolio list
        user_ids = np.array(df.index)
        portfolio_ids = np.array(df.columns)

        row = np.where(user_ids == user_id)[0][0]
        col = np.where(portfolio_ids == offer_id)[0][0]

        # Now we take the dot product of that row and column in to make prediction
        pred = np.dot(user_mat[row, :], portfolio_mat[:, col])

        return pred  # return the prediction here

    except Exception as e:
        print(e)
        return None


def evaluate_model(test_user_item_matrix, user_mat, portfolio_mat):
    """
    We use this to test accuracy of our predictions by measuring
    the sum of squared errors.
    """
    n = np.count_nonzero(~np.isnan(test_user_item_matrix))

    # keep track of the sum of squares
    sse = 0

    for user_id in test_user_item_matrix.index:
        for offer_id in test_user_item_matrix.columns.values:
            if ~np.isnan(test_user_item_matrix.loc[user_id, offer_id]):
                pred = predict(test_user_item_matrix, user_mat, portfolio_mat, user_id, offer_id)
                if pred:
                    diff = test_user_item_matrix.loc[user_id, offer_id] - pred
                    sse += diff ** 2
    return sse / n


def make_offer(df, df_portfolio, user_mat, portfolio_mat, user_id):
    """
    Recommend an offer to a user
    """
    offer_bucket = {}  # A list of candidate offers for this user

    # Make some predictions here
    for offer_id in df.columns:
        pred = predict(df, user_mat, portfolio_mat, user_id, offer_id)
        if pred:
            offer_bucket[offer_id] = [pred, df_portfolio[df_portfolio['id'] == offer_id]]
    return sorted(offer_bucket.items(), key=lambda k: (k[1], k[0]), reverse=True)

def main():

    if len(sys.argv) == 3:

        print('\n[ info ] {}'.format('Starbucks Capstone Project'))
        print('--------------------------------------------------------')
        print('\n[ info ] {}'.format('Building Recommender System ...'))

        # path to files
        database_filepath, model_filepath = sys.argv[1:]
        database_filepath = os.path.join(os.getcwd(), os.path.join('data', database_filepath))
        transcript_filepath = os.path.join(os.getcwd(), os.path.join('data', 'transcript.db'))
        portfolio_filepath = os.path.join(os.getcwd(), os.path.join('data', 'portfolio.db'))
        model_filepath = os.path.join(os.getcwd(), os.path.join('models', model_filepath))

        print('\n[ info ] Loading data...\n\t df: {} \n\t transcript: {} \n\t portfolio: {}'
              .format(database_filepath, transcript_filepath, portfolio_filepath))
        df = load_data(database_filepath, 'df')
        transcript = load_data(transcript_filepath, 'transcript')
        portfolio = load_data(portfolio_filepath, 'portfolio')

        print('\n[ info ] Fit our model using FunkSVD...')
        train_df, test_df, user_item_train, user_item_test, test_idx, test_offers, user_mat, portfolio_mat, mse = fit(transcript,
                                                                                                                      portfolio)

        print('\n[ info ] Mean Squared Error: {}'.format(mse))

        #latent_features = [k for k in range(1, 51, 2)]
        #test_results = {}
        #for k in latent_features:
        #    test_results[k] = FunkSVD(user_item_train, latent_features=k, learning_rate=0.0001, iters=250)
        #    print(k, test_results[k][2])

        #validation_set_results = {}
        #for k in latent_features:
        #    validation_set_results[k] = evaluate_model(user_item_test, test_results[k][0], test_results[k][1])
        #    print(k, validation_set_results[k])
        print('\n[ info ] Evaluating model... using 27 latent features ...')
        # I am going to use 27 latent features, obtain from above run.
        print('\n[ info ] SSE', evaluate_model(user_item_test, user_mat, portfolio_mat))

        #print('\n[ info ] best validation accuracy: {}'.format(max(validation_set_results, key=validation_set_results.get)))
        # 39 latent features produced the best validatiom accuracy results
        user_id = '0009655768c64bdeb2e877511632db8f'
        offer_id = '0b1e1539f2cc45b7b9fa7c272da2e1d7'

        print('\n[ info ] Making a recommendation for a user: {}'.format(user_id))
        recs = make_offer(train_df, portfolio, user_mat, portfolio_mat, user_id)
        print('\n[ info ] Recommendations for user are: \n{}'.format(recs))

    else:
        print('Please provide the filepath of the Starbucks database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'python -m models.train_classifier starbucks.db classifier.pkl')

if __name__ == '__main__':
    main()
