"""
Currently uses a pretrained GloVe model to embed words into 300d vectors.
"""

from os import path as ospath

dirname, _ = ospath.split(ospath.abspath(__file__))
GLOVE_PATH = dirname + '/../data/glove.txt'

def loadGlove(path):
    """
    Loads the pretrained glove model into memory.

    Arguments:
      - path to the `glove.txt` file.

    Returns:
      - Dictionary with words as key and vector as value.
    """
    print("Starting to import GloVe\n")
    f = open(path, 'r')
    model = {}

    for i, line in enumerate(f):
        if i % 10000 == 0:
            print(str(i) + '/2196017\r')
        splitLine = line.split(" ")
        word = splitLine[0]

        embedding = [float(val) for val in splitLine[1:]]
        model[word] = embedding

    print("Finished importing")

    return model

def embed_word(word):
    """
    Given a word, return embedding or throws a `KeyError` exception.
    """
    return glove[word]

glove = loadGlove(GLOVE_PATH)
