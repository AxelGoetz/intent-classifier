"""
Find URLs in a text.
"""

from re import findall

url_regex = "((https?://)?[^\s]+\.[^\s]+)"


def get_urls(sentence):
    """
    Gets an array of urls from the tokens.

    Parameters:
      - sentence: A string representing a sentence.
    """
    return [x[0] for x in findall(url_regex, sentence)]
