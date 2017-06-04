"""
Extract date and/or time entities
"""

import datefinder

def get_dates(sentence):
    """
    Gets the date from the list of tokens.

    Parameters:
      - sentence: A string, representing a sentence.

    Returns:
      - An array of datetime objects.
    """
    # TODO: Currently uses `datefinder` but need to find a more permanent solution.
    matches = datefinder.find_dates(sentence)
    return [match for match in matches]
