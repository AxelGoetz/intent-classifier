"""
Pickle and unpickle python objects.
"""

def save_object(obj, file_name):
    """
    Used to save an object to a file

    @param obj is the object you want to save
    @param file_name is the name that you are saving the object to *(Should have a `.pkl` extension)*
    """
    from pickle import dump

    with open(file_name, "wb") as f:
        dump(obj, f) # Use the default protocol

def load_object(file_name):
    """
    Loads an object from a file

    @param file_name is the file name of the file *(with the `.pkl` extension)
    """
    from pickle import load

    with open(file_name, "rb") as f:
        return load(f)
