import sys
import pandas as pd
from sqlalchemy import create_engine
import logging
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import pickle
import nltk
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
current_time = datetime.now().strftime("%Y%m%d%H%M%S") # get current time
nltk.download(['punkt_tab', 'wordnet'])


def load_data(database_filepath: str) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """ Load the data from the database

    Args:
        database_filepath (str): Path to the database file

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, List[str]]: Tuple of input data, labels, and category names
    """
    # Load the data
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('main', engine).dropna()
    logging.info("Data loaded")
    logging.debug(f"Data shape: {df.shape}")

    # Get the input and labels
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = Y.columns

    return X, Y, category_names


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
    tokens = word_tokenize(text)
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token).strip() for token in tokens]
    # Filter out stop words and punctuation
    tokens = list(filter(lambda x: x.isalnum(), tokens))
    
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = list(filter(lambda x: x not in stop_words, tokens))
    logging.debug(f"Tokens: {tokens}")

    return tokens
    


def build_model():
    """ Build the model using a pipeline and grid search

    Returns:
        GridSearchCV: Model that has been trained
    """
    # Create the pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
        ('clf', MultiOutputClassifier(XGBClassifier()))
    ])
    logging.info("Pipeline created")
    logging.debug(f"Pipeline: {pipeline}")

    # Set the parameters for the grid search
    parameters = {
        'tfidf__max_df': (0.5, 0.75, 1.0),
        'tfidf__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__max_depth': [3, 5, 7, 9]
    }
    logging.info("Parameters set")
    logging.debug(f"Parameters: {parameters}")

    # Create the grid search object
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3, n_jobs=-1)

    return cv


def evaluate_model(model: GridSearchCV, X_test: pd.DataFrame,
                   Y_test: pd.DataFrame, category_names: List[str]) -> None:
    """ Evaluate the model using the test data and print the classification report

    Args:
        model (GridSearchCV): Grid search model
        X_test (pd.DataFrame): Test input data
        Y_test (pd.DataFrame): Test labels
        category_names (List[str]): List of category names
    """
    assert len(category_names) == Y_test.shape[1], "Number of category names must match number of columns in Y_test"

    # Predict the labels
    Y_pred = model.predict(X_test)
    logging.info("Labels predicted")
    logging.debug(f"Predicted labels: {Y_pred}")

    # Print the classification report
    for i, category in enumerate(category_names):
        print(f"\n------------------Category: {category}------------------")
        print(classification_report(Y_test.iloc[:, i], Y_pred[:, i]))
        print("------------------------------------------------------")

    # Save classification report to a file
    with open(f"classification_report_{current_time}.txt", 'w') as file:
        for i, category in enumerate(category_names):
            file.write(f"\n------------------Category: {category}------------------\n")
            file.write(classification_report(Y_test.iloc[:, i], Y_pred[:, i]))
            file.write("\n------------------------------------------------------\n")

    logging.info("Classification report printed")


def save_model(model: GridSearchCV, model_filepath: str) -> None:
    """ Save the model to a pickle file

    Args:
        model (GridSearchCV): Model to save
        model_filepath (str): Path to save the model
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)
    logging.info("Model saved")
    logging.debug(f"Model saved to: {model_filepath}")


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
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