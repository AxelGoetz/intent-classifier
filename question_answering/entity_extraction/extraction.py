"""
Extracts entities from raw text.
Currently extracts the following:
  - Location
  - Organization
  - Person
  - Duration
  - Data
  - Cardinal
  - Percent
  - Money
  - Measure
  - Color
"""

from nltk.chunk import ne_chunk
from nltk.tree import Tree
from nltk import pos_tag

from date_time import get_dates
from colors import get_colors
from url import get_urls

def extraction(tokens):
    """
    Performs entity extraction.

    There are two main data structures, each with their own advantage that
    can be returned:
      1. A dictionary with as key the entity name, and as value an array of entity values.
        (e.g. {color: [green]}).
      2. An array with the same length as the input array. Each index of the array contains a list
        containing the entities that instance matched (e.g. ['I', 'am', 'Axel'] → [['subject'], ['verb'], ['Name', 'adverb']).

    At the moment, we choose the first option but the code should not be too difficult to change.

    Parameters:
      - tokens: An array of strings/tokens (preferably not cleaned).

    Returns:
      - A dictionary in the format `{entity: [tokens]}`.
    """
    tagged = pos_tag(tokens)

    entities = nltk_extraction(tagged)
    entities['color'] = get_colors(tokens)
    entities['datetime'] = get_dates(tokens)
    entities['url'] = get_urls(tokens)


def nltk_extraction(tagged):
    """
    Performs the prebuild nltk entity extraction.
    Recognizes location, organization, person, duration, data, cardinal, percent, money, measure.

    Parameters:
      - tagged: An array of tagged strings/tokens in the format `[(tag, token)]`.

    Returns:
      - A dictionary `{entity: [tokens]}`.
    """
    entities = {}
    parse_tree = ne_chunk(tagged)

    for child in parse_tree:
        if isinstance(child, Tree):
            string = " ".join(child.leaves)
            label = child.label.lower()

            if label not in entities:
                entities[label] = []
            entities[label].append(string)

    return entities
