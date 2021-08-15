import json
import os
import plotly
import pandas as pd
import nltk
import pickle 
import joblib
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from datetime import datetime

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sqlalchemy import create_engine

from . import utils

app = Flask(__name__)

# load data

print('() [ info ] Opening database using engine ...'.format(datetime.now()))
db_path = 'data/DisasterResponse.db'

print('{} [ info ] DB path: {}'.format(datetime.now(), db_path))

pkl_path = "models/classifier.pkl"

print('{} [ info ] Pickle path: {}'.format(datetime.now(), pkl_path))
engine = create_engine('sqlite:///{}'.format(db_path))
df = pd.read_sql_table('message', engine)

# load model
model = joblib.load(pkl_path)

print('{} [ info ] Engine running, dataframe created and model loaded ... we are in business.'.format(datetime.now()))

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

# comment this out for Heroku deployment
def main():
    app.run(host='0.0.0.0', port=3001, debug=True)

# comment this out for Heroku deployment
if __name__ == '__main__':
    main()
