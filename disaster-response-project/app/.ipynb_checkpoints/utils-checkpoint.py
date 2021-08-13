import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def tokenize(text):
    """
    tokenization function to process text data.
    params
    -----------------------------------------------------------
    INPUT
    text: text to tokenize
    OUTPUT
    clean_tokens: tokenized and cleaned text
    """

    # get tokens here
    tokens = word_tokenize(text)

    # instance of our lemmatizer
    lemmatizer = WordNetLemmatizer()

    # clean tokens here
    clean_tokens = [lemmatizer.lemmatize(token).lower().strip()
                    for token in tokens if token.isalpha()]

    # return clean tokens
    return clean_tokens