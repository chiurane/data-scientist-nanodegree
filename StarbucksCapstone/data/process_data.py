import sys
import os
import pandas as pd
import numpy as np
import datetime
import json
import datetime
from sqlalchemy import create_engine

def load_data(portfolio_filepath, profile_filepath, transcript_filepath):
    """
    Loads data from the three dataset files
    params
    -------------------------------------------------------------
    :param portfolio_filepath: path to portfolio file
    :param profile_filepath: path to profile fle
    :param transcript_filepath: path to transcript file
    :return: three dataframes for each of the files
    """
    try:
        # Read in the portfolio json file
        portfolio = pd.read_json(portfolio_filepath, orient = 'records', lines = True)

        # Read in the profile json file
        profile = pd.read_json(profile_filepath, orient = 'records', lines = True)

        # Read in the transcript json file
        transcript = pd.read_json(transcript_filepath, orient = 'records', lines = True)

        # A dictionary of dataframes
        df = {'portfolio': portfolio,
              'profile':profile,
              'transcript': transcript}

        return df
    except Exception as e:
        print('{} [ info ] An error has occured with the following details: {}'.format(datetime.now(), e))


def clean_portfolio(portfolio):
    """
    cleans up our portfolio dataset by:
    - performing one-hot encoding on the offer column
    - performing one-hot encoding on the channels column
    - concaternating the two datasets into a single one
    params
    ---------------------------------------------------------------
    :param portfolio - the portfolio dataframe
    :return clean_portfolio - the the cleaned up portfolio
    """
    # if either offer_type or channels are not in the dataset then
    # we assume portfolio is clean already
    # if not 'offer_type' in portfolio.columns or 'channels' in portfolio.columns:
    #    return portfolio

    portfolio['offer_type_dummy'] = portfolio['offer_type']

    # one-hot encoding on offer_type column
    clean_portfolio = pd.get_dummies(portfolio, columns=['offer_type'])
    clean_portfolio = clean_portfolio.rename(columns={'offer_type_dummy': 'offer_type'})

    # one-hot encoding on channels column
    channels = list(clean_portfolio['channels'])
    channels_df = pd.DataFrame({'channels': channels})
    channels_df = pd.get_dummies(channels_df.channels.apply(pd.Series).stack()).sum(level=0)

    # concate the dataframes here
    clean_portfolio = pd.concat([clean_portfolio, channels_df], axis=1)

    # drop the channels column
    drop_cols = ['channels']
    clean_portfolio.drop(columns=drop_cols, inplace=True)

    # return the clean_portfolio
    return clean_portfolio

def calculate_member_days(df):
    """
    calculate the number of days a profile has been a member.
    params
    -----------------------------------------------------------------
    param: df - the dataframe with member information
    return: new df with member_since_days added
    """
    import datetime
    # compute the number of days a profile has been a member
    df['member_since_days'] = datetime.datetime.today().date() - pd.to_datetime(df['became_member_on'],
                                                                                format='%Y%m%d').dt.date

    # remove the trailing 'days'format from it
    df['member_since_days'] = df['member_since_days'] / np.timedelta64(1, 'D')

    return df


def clean_profile(df):
    """
    clean profile
    params
    ---------------------------------------------------------------------
    param: df the profile dataset
    return df cleaned profiles
    """
    # calculate member days first
    df = calculate_member_days(df)

    # lets replace the None first
    df.replace(to_replace=[118], value=np.nan, inplace=True)
    df.replace(to_replace=[None], value=np.nan, inplace=True)

    # now the nulls can go
    df = df[df['gender'].notna()]

    return df # return the clean profile


# now we create a function to perform this mapping in our pipeline
def events_to_outcomes(df):
    """
    maps events to events outcomes so that we can use this later when reframe
    this problem as a classification problem
    params
    --------------------------------------------------------------------
    param: df dataframe of transripts
    return: datafram with mapping of events to outcomes
    """
    # first we extract our events here
    unq_events = df['event'].unique()
    unq_events = pd.Series(unq_events).to_dict()

    # lets swap around the keys and values so we map in the dataframe
    unq_events = dict([(value, key) for key, value in unq_events.items()])

    # now for the magic
    df['outcome'] = df['event'].map(unq_events)

    # return our new data_frame
    return df


def clean_transcript(df):
    """
    cleans the transcript dataframe
    params
    --------------------------------------------------------------
    param: df the transcript data
    return df clean dataframe
    """
    def clean_funky(df):
        df = df[
            df['value'].apply(lambda x: True if ('offer id' in x) or ('offer_id' in x) else False)]
        df['offer_id'] = df['value'].apply(
            lambda x: x['offer id'] if ('offer id' in x) else x['offer_id'])
        return df
    df = clean_funky(df)
    df = df[['person', 'event', 'offer_id']]
    # 1) first we do our event to outcomes mapping
    #df = events_to_outcomes(df)

    # 2) Normalize values
    #normalized_values = pd.json_normalize(df['value'])
    #normalized_values['offer_id'] = normalized_values['offer id'][normalized_values['offer_id'].isnull()]

    # 3) Drop offer id column
    #drop_cols = ['offer id']
    #normalized_values.drop(columns=drop_cols, inplace=True)

    # 4) merge normalized values dataframe back into transcript dataframe
    #df = pd.concat([df, normalized_values], axis=1)

    # 5) drop the value column
    #drop_cols = ['value']
    #df.drop(columns=drop_cols, inplace=True)

    # 6) One-hot encoding for event column
    #df['dummy_event'] = df['event'] # I am retaining this for FunkSVD later
    #df = pd.get_dummies(df, columns=['event'])
    #df = df.rename(columns={'event_offer completed': 'offer completed',
    #                        'event_offer received': 'offer received',
    #                        'event_offer viewed': 'offer viewed',
    #                        'event_transaction': 'transaction',
    #                        'dummy_event':'event'})

    # 7) return our new clean data frame
    #print(df.shape)
    return df # return our clean transcript here

def data_cleaning_pipeline(df_portfolio, df_profile, df_transcript, database_filepath):
    """
    perform all our data cleaning functions in a pipeline and return a single dataset for processing and saving to some db
    """

    # clean portfolio
    df_portfolio = clean_portfolio(df_portfolio)
    portfolio_filepath = os.path.join(os.getcwd(), os.path.join('data', 'portfolio.db'))
    print('\n[ info ] Saving portfolio to database {}'.format(portfolio_filepath))
    save_data(df_portfolio, portfolio_filepath, 'portfolio')

    # clean profile
    # df_profile = calculate_member_days(df_profile)
    df_profile = clean_profile(df_profile)
    profile_filepath = os.path.join(os.getcwd(), os.path.join('data', 'profile.db'))
    print('\n[ info ] Saving profile to database {}'.format(profile_filepath))
    save_data(df_profile, profile_filepath, 'profile')

    # clean transcript
    df_transcript = clean_transcript(df_transcript)
    transcript_filepath = os.path.join(os.getcwd(), os.path.join('data', 'transcript.db'))
    print('\n[ info ] Saving transcript to database {}'.format(transcript_filepath))
    save_data(df_transcript, transcript_filepath, 'transcript')

    # merge all datasets into a single one
    df = pd.merge(df_transcript, df_portfolio, how='left', left_on='offer_id', right_on='id')
    drop_cols = ['id']
    df.drop(columns=drop_cols, inplace=True)

    df = pd.merge(df_profile, df, how='right', left_on='id', right_on='person')

    # lets remove transactions where biographic information is not present
    df = df[df['gender'].notna()]
    df.drop(columns=drop_cols, inplace=True)

    # return our new merged df
    return df

def save_data(df, db_filename, table_name):
    """"
    INPUT:
    df - pandas dataframe to save to database
    df_filename - database file name
    OUTPUT:

    Description:
    saves our df to the database
    """

    # first check if the database exists
    if os.path.exists(db_filename):
        os.remove(db_filename)

    # now save
    engine = create_engine('sqlite:///{}'.format(db_filename))
    df.to_sql(table_name, engine, index = False, if_exists='replace')

def main():
    print('\n[ info ] {}'.format('Starbucks Capstone Project'))
    print('--------------------------------------------------------')
    print('\n[ info ] {}'.format('Performing data cleaning and engineering ...'))
    if len(sys.argv) == 5:
        # file paths
        portfolio_filepath, profile_filepath, transcript_filepath, database_filepath = sys.argv[1:]
        portfolio_filepath = os.path.join(os.getcwd(), os.path.join('data', portfolio_filepath))
        profile_filepath = os.path.join(os.getcwd(), os.path.join('data', profile_filepath))
        transcript_filepath = os.path.join(os.getcwd(), os.path.join('data', transcript_filepath))
        database_filepath = os.path.join(os.getcwd(), os.path.join('data', database_filepath))

        print('\n[ info ] Loading data ...\n\t Portfolio: {} \n\t Profile: {} \n\t Transcript: {}'
              .format(portfolio_filepath, profile_filepath, transcript_filepath))

        # load data
        df = load_data(portfolio_filepath, profile_filepath, transcript_filepath)
        df_portfolio, df_profile, df_transcript = df['portfolio'], df['profile'], df['transcript']

        # data engineering and cleaning here
        print('\n[ info ] Cleaning data ... \n\t Portfolio dataset: {} \n\t Profile dataset: {} \n\t Trascript dataset: {}'
              .format(df_portfolio.shape, df_profile.shape, df_transcript.shape))

        # Save the full df here
        df = data_cleaning_pipeline(df_portfolio, df_profile, df_transcript, database_filepath)
        print('\n[ info ] Saving df to database .... \n\t Starbucks database: {}'.format(database_filepath))
        save_data(df, database_filepath, 'df')

        print('\n[ info ] All data cleaning and feature engineering tasks completed.')

if __name__ == '__main__':
    main()

