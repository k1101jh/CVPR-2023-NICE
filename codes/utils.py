import io
import json
import numpy as np
import matplotlib.pyplot as plt
import string


def show_image_caption(image, caption, save_path=None, show_fig=False):
    # title_string = '\n'.join([caption[0] for caption in captions])
    fig = plt.imshow(image)
    plt.title(caption)
    
    if save_path is not None:
        # plt.ioff()
        plt.savefig(save_path)
    
    if show_fig:
        plt.show()
        
    return fig

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