import os
import random
import pandas as pd
# from PIL import Image
from typing import List, Any

import torch
from torch.utils import data

from datasets import load_dataset
from datasets import Dataset, Image


class NICETrainDataset(data.Dataset):
    def __init__(self, img_dir, ann_file, processor, vis_processor, tokenizer):
        super().__init__()
        self.img_dir = img_dir
        self.ann_file = ann_file
        self.processor = processor
        self.vis_processor = vis_processor
        self.tokenizer = tokenizer
        self.prompt = "a photo of "
        self.max_length = 50
        train_df = pd.read_csv(self.ann_file)
        self.caption_ds = Dataset.from_pandas(train_df)

        # image ds
        self.image_filename_list = self.caption_ds['public_id']
        self.image_path_list = [os.path.join(
            self.img_dir, str(image_filename) + '.jpg') for image_filename in self.image_filename_list]
        self.image_ds = Dataset.from_dict(
            {'image': self.image_path_list}).cast_column("image", Image())

        # caption ds
        self.caption_ds = self.caption_ds.map(
            self.train_batch_preprocess,
            # remove_columns=['public_id', 'caption_gt', 'category'],
            batched=True,
            batch_size=1000,
        )
        
        # prompt to tokens
        # self.input_tokens = self.tokenizer(
        #     self.prompt, padding='max_length', max_length=self.max_length, return_tensors='pt'
        # )

    def train_batch_preprocess(self, batch):
        """ 여기서 prompt와 caption을 합치고 token으로 만든다.
        원본에는 pre_caption 함수를 정의하여 몇몇 문자를 치환한다.
        (lavis/processors/blip_processors.py 참고)

        Args:
            batch (_type_): batch 단위 입력

        Returns:
            torch.tensors: prompt + caption을 token으로 변환한 결과
        """
        target_text = batch['caption_gt']
        # target_text = [self.prompt + t for t in target_text]
        
        target = self.tokenizer(
            target_text, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return target

    def __getitem__(self, x):
        # return self.processor(image=self.image_ds[x]['image'], text=self.prompt + self.caption_ds[x]['caption_gt'], return_tensors="pt")
        image = self.vis_processor(
            self.image_ds[x]['image'], return_tensors="pt")
        image['pixel_values'] = image['pixel_values'].squeeze(0).to(torch.float16)
        caption = self.caption_ds[x]

        return image['pixel_values'], caption['input_ids'], caption['attention_mask']

    def __len__(self):
        return len(self.caption_ds)


class NICETestDataset(data.Dataset):
    def __init__(self, img_dir, vis_processor):
        super().__init__()
        self.img_dir = img_dir
        self.vis_processor = vis_processor
        self.image_filename_list = os.listdir(self.img_dir)
        self.image_path_list = [os.path.join(
            self.img_dir, image_filename) for image_filename in self.image_filename_list]
        self.ds = Dataset.from_dict(
            {'image': self.image_path_list}).cast_column("image", Image())

    def __getitem__(self, x):
        image = self.vis_processor(self.ds[x]['image'], return_tensors="pt")
        image['pixel_values'] = image['pixel_values'].squeeze(0)
        filename = self.image_filename_list[x].split('.')[0]
        return image, filename

    def __len__(self):
        return len(self.ds)
