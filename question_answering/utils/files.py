"""
Reading and manipulating files.
"""

import json

from os import scandir, makedirs, path as ospath

def get_files(data_dir, extension):
    """
    Gets all the files in a directory.

    Parameters:
      - data_dir: A string containing the absolute path to the data directory.
      - extension: Extension of all the files in the `data_dir` (including the '.').
    """
    paths = []
    for i, f in enumerate(scandir(data_dir)):
        if f.is_file() and f.name[-len(extension):] == extension:
            paths.append(f.path)

    return paths

def open_file(path):
    """
    Opens a file and returns the necessary data.

    Parameters:
      - path: Absolute path to file.
    """
    with open(path, 'r') as f:
        return json.load(f)

def create_dir_if_not_exists(directory):
    """
    Creates a dir if it does not exists, if it exists, it deletes it and creates a new one

    Parameters:
      - directory: A string containing the **absolute** path to the directory
    """
    if ospath.exists(directory):
        from shutil import rmtree
        rmtree(directory)

    makedirs(directory)

def get_name(path):
    """
    Extract a name (without extension) from a path.

    Arguments:
      - path: A string, representing a path.
    """
    file_name = ospath.basename(path)
    return ospath.splitext(file_name)[0]
