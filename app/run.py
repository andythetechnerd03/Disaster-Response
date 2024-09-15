import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine
from typing import List, Dict
import nltk
from wordcloud import WordCloud
nltk.download(['punkt', 'wordnet', 'stopwords'])
import logging
import os
logging.basicConfig(level=logging.INFO)
from collections import Counter

# import statements
app = Flask(__name__, static_folder='static')

def tokenize(text: str) -> List[str]:
    """ Tokenize the text by converting to lowercase, tokenizing, and lemmatizing

    Args:
        text (str): Text to tokenize

    Returns:
        List[str]: List of tokens that have been lemmatized and converted to lowercase.
    """
    # Convert to lowercase
    text = text.lower()
    # Tokenize
    tokens: List = word_tokenize(text)
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token).strip() for token in tokens]
    # Filter out stop words and punctuation
    tokens = list(filter(lambda x: x.isalnum(), tokens))
    
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = list(filter(lambda x: x not in stop_words, tokens))
    logging.debug(f"Tokens: {tokens}")

    return tokens

# load data
engine = create_engine('sqlite:///../data/Disaster.db')
df = pd.read_sql_table('main', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    if not os.path.exists('static/bigrams.json'):
        bigram_counts = get_bigrams_count(df)
        top_10_bigrams = dict(sorted(bigram_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        with open('static/bigrams.json', 'w') as f:
            json.dump(top_10_bigrams, f)
    else:
        with open('static/bigrams.json', 'r') as f:
            top_10_bigrams = json.load(f)

        
    logging.debug(f"Top 10 bigrams: {top_10_bigrams}")
    
    # # create visuals
    # # TODO: Below is an example - modify to create your own visuals

    graphs = [
        {
            'data': [
                Bar(
                    x=list(top_10_bigrams.keys()),
                    y=list(top_10_bigrams.values())
                )
            ],

            'layout': {
                'title': 'Top 10 Bigrams (Two-word phrases) in training data',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Bigram"
                }
            }
        }
    ]

    ## generate word cloud then render template
    if not os.path.exists('static/wordcloud.png'):
        print("Generating wordcloud")
        generate_wordcloud(df)
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly 
    return render_template('master.html', user_image='static/wordcloud.png', graphJSON=graphJSON, ids=ids)

def generate_wordcloud(df: pd.DataFrame) -> None:
    """ Generate a word cloud from the dataframe

    Args:
        df (pd.DataFrame): Dataframe containing the messages
    """
    text_list = df['message'].tolist()
    tokens = []
    for text in text_list:
        tokens.extend(tokenize(text))
    full_text = ' '.join(tokens)
    logging.info("Wordcloud generated")
    logging.debug(f"Wordcloud text: {full_text}")
    wordcloud = WordCloud().generate(full_text)
    wordcloud.to_file('static/wordcloud.png')
    assert os.path.exists('static/wordcloud.png'), "Wordcloud not generated"

def get_bigrams_count(df: pd.DataFrame) -> Dict[str, int]:
    """ Get the bigrams from the dataframe

    Args:
        df (pd.DataFrame): Dataframe containing the messages

    Returns:
        Dict[str, int]: Dictionary of bigrams and their counts
    """
    text_list = df['message'].tolist()
    tokens = []
    for text in text_list:
        tokens.extend(tokenize(text))
    bigrams = list(map(lambda x: ' '.join(x), nltk.bigrams(tokens)))
    bigram_counts = Counter(bigrams)
    return bigram_counts

# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification = model.predict_proba([query])
    classification_labels = list(map(lambda x: x.argmax(), classification))
    classification_probs = list(map(lambda x: x[0][1].item(), classification))
    print(len(df.columns[4:]), len(classification_probs))
    classification_results = dict(zip(df.columns[4:], classification_probs))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()