import os
from utils import load_annotations


if __name__ == '__main__':
    
    COCO_dataset_config = {
        "train_folder": "../COCO/images/train2017",
        "val_folder": "../COCO/images/val2017",
        "test_folder": "../COCO/images/test2017",
        "annotation_file_path": "../COCO/annotations/captions_train2017.json",
    }
    
    annotations = load_annotations(COCO_dataset_config["annotation_file_path"])
    print(annotations)