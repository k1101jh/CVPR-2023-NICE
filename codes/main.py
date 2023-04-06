import os
from utils import load_annotations
from dataset_config import COCO_dataset_config


if __name__ == '__main__':
    annotations = load_annotations(COCO_dataset_config["annotation_file_path"])
    print(annotations)