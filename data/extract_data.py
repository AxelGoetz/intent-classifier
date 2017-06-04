"""
Preprocesses the data in the `original_data` directory such that it can easily be used for training.
"""

import json

from os import path as ospath

dirname, _ = ospath.split(ospath.abspath(__file__))
DATA_DIR = dirname + '/original_data'

from sys import path

# Hack to import from sibling directory
path.append(ospath.dirname(path[0] + '/../question_answering/'))

from utils import files, clean_text, word_embedding

def extract_data(path):
    """
    Given the absolute path, extracts the json data.

    Return:
      - A dictionary `{intent, data: [{keys: value}]}`.
    """
    with open(path, 'r') as f:
        json_data = json.load(f)

        intent = json_data["intent"]
        new_data = []

        for row in json_data['data']:
            new_row = {}
            new_row['raw_text'] = row['raw_text']
            new_row['tokens'] = clean_text.clean_text(row['raw_text'])
            new_row['embeddings'] = []

            for token in new_row['tokens']:
                try:
                    embedding = word_embedding.embed_word(token)
                    new_row['embeddings'].append(embedding)
                except KeyError:
                    pass

            new_data.append(new_row)


        return {'intent': intent, 'data': new_data}

def write_json(name, data, directory):
    """
    Given an intent and data, writes to json file.

    Parameters:
      - name: String (without '.json').
      - data: A dictionary `{intent, data: [{keys: value}]}`.
      - directory: The absolute path to the directory to write the file to.
    """
    with open(directory + name + '.json', 'w') as f:
        f.write(json.dumps(data))

def main():
    """
    Extracts the data from the `apiai_data` directory and puts it in a `classifier_data` directory.
    """
    paths = files.get_files(DATA_DIR, ".json")

    csv_dir = dirname + '/classifier_data/'
    files.create_dir_if_not_exists(csv_dir)

    for path in paths:
        data = extract_data(path)
        write_json(data['intent'], data, csv_dir)

if __name__ == '__main__':
    main()
