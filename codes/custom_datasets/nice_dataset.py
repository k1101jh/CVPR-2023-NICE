import os
import random
# from PIL import Image
from typing import List, Any
from torch.utils import data
# from torch.utils.data import Dataset

from datasets import load_dataset
from datasets import Dataset, Image


# class NICEValDataset(Dataset):
#     def __init__(self, img_dir, ann_file, vis_processor):
#         super(Dataset, self).__init__()
#         self.img_dir = img_dir
#         self.ann_file = ann_file
#         self.vis_processor = vis_processor
#         self.caption_ds = load_dataset("csv", data_dir=self.ann_file)
#         print(self.caption_ds)

    
class NICETestDataset(data.Dataset):
    def __init__(self, img_dir, vis_processor):
        super().__init__()
        self.img_dir = img_dir
        self.vis_processor = vis_processor
        self.image_filename_list = os.listdir(self.img_dir)
        self.image_path_list = [os.path.join(self.img_dir, image_filename) for image_filename in self.image_filename_list]
        self.ds = Dataset.from_dict({'image': self.image_path_list}).cast_column("image", Image())
        
    def __getitem__(self, x):
        input = self.vis_processor(self.ds[x]['image'], return_tensors="pt")
        input['pixel_values'] = input['pixel_values'].squeeze(0)
        label = self.image_filename_list[x].split('.')[0]
        return input, label
    
    def __len__(self):
        return len(self.ds)