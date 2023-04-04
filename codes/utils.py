import json


def load_annotations(json_path):
    with open(json_path) as f:
        json_object = json.load(f)
        
        annotations = json_object["annotations"]
    
    return annotations