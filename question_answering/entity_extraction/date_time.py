"""
Extract date and/or time entities
"""

import datefinder

def get_dates(tokens):
    """
    Gets the date from the list of tokens.

    Parameters:
      - tokens: An array of strings/tokens.

    Returns:
      - An array of datetime objects.
    """
    # TODO: Currently uses `datefinder` but need to find a more permanent solution.
    matches = datefinder.find_dates(string_with_dates)
    return [match for match in matches]
