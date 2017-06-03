"""
Cleaning raw sentence data.
"""

# TODO: Should we stem/lemmatize?
# TODO: Paired words (New_York)?

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation

import re

def tokenize(sentence):
    """
    Returns tokens
    """
    return word_tokenize(sentence)

def remove_values(tokens, exclude):
    """
    Removes the characters in the remove string from all the tokens

    Arguments:
      - tokens: Array of strings.
      - remove: Set of characters, to be removed
    """
    new_tokens = []
    for token in tokens:
        new_token = s = ''.join(char for char in token if char not in exclude)
        if not new_token == '':
            new_tokens.append(new_token)

    return new_tokens

def remove_punctuation(tokens):
    """
    Removes all puntuation in all tokens.
    """
    exclude = set(punctuation)

    # TODO: Add more
    exclude.update(['\b', '\n', '\t'])

    return remove_values(tokens, exclude)

def remove_numbers(tokens):
    """
    Removes all numbers from tokens.
    """
    exclude = set("1234567890")
    return remove_values(tokens, exclude)


def remove_stopwords(tokens):
    """
    Removes words that don't really contribute to the meaning of the sentence.
    This makes the sequences shorter and allows for faster learning.

    Arguments:
      - words: Array of strings, representing tokens
    """
    new_tokens = []
    for token in tokens:
        if not token in stopwords.words('english'):
            new_tokens.append(token)

    return new_tokens

def clean_text(sentence):
    """
    Returns a list of tokens, where punctuation and stopwords are removed.

    Arguments:
      - sentence: String, which represents a sentence.

     Returns:
       - List of strings (tokens).
    """
    tokens = tokenize(sentence)
    tokens = remove_punctuation(tokens)
    tokens = remove_numbers(tokens)
    tokens = remove_stopwords(tokens)
    tokens = [s.lower() for s in tokens]

    return tokens
