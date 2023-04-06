import json


def load_annotations(json_path):
    with open(json_path) as f:
        json_object = json.load(f)
        
        annotations = json_object["annotations"]
    
    return annotations

def get_YN_answer(base_str, default=None):
    """ Get Yes or No input from user

    Args:
        default (Iterable, optional): Treats user's ENTER input as this value . Defaults to None.

    Returns:
        bool: if user input is Y, return True. else if user input is N, return False.
    """
    assert(default in ["Y", "y", "N", "n", None])
    y_inputs = ["Y", "y"]
    n_inputs = ["N", "n"]
    
    if default in y_inputs:
        string_to_print = "(Y/n)>"
    elif default in n_inputs:
        string_to_print = "(y/N)>"
    else:
        string_to_print = "(y/n)>"
    
    while True:
        inp = input(base_str + string_to_print)
        if inp == "":
            inp = default
        if inp in y_inputs:
            return True
        elif inp in n_inputs:
            return True