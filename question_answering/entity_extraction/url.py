"""
Find URLs in a text.
"""

from re import findall

url_regex = "((https?://)?[^\s]+\.[^\s]+)"


def get_urls(tokens):
    """
    Gets an array of urls from the tokens.

    Parameters:
      - tokens: A list of strings/tokens.
    """
    return [x[0] for x in findall(url_regex, " ".join(tokens))]
