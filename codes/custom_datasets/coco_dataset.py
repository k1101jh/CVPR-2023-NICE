# https://github.com/pytorch/vision/blob/main/torchvision/datasets/coco.py
# https://github.com/rammyram/image_captioning/blob/master/Image_Captioning.ipynb
import os
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
from typing import List, Tuple, Any
from tqdm import tqdm
import random


class COCOBaseDataset(Dataset):
    def __init__(self, img_dir, ann_file, vis_processor, text_processor):
        super(Dataset, self).__init__()
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.ids = list(sorted(self.coco.imgs.keys()))
        
    def _load_image(self, id):
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.img_dir, path)).convert("RGB")

    def _load_target(self, id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))
    
    def __getitem__(self, index):
        id = self.ids[index]
        image = self._load_image(id)
        ann = self._load_target(id)

        return {
            "image": self.vis_processor(image),
            "text_input": self.text_processor(ann[random.randrange(len(ann))]["caption"]),
            "image_id": id
        }
    
    def __len__(self) -> int:
        return len(self.ids)  


class COCOTrainDataset(COCOBaseDataset):
    def __init__(self, img_dir, ann_file, vis_processor, text_processor):
        super(COCOTrainDataset, self).__init__(img_dir, ann_file, vis_processor, text_processor)

    def __getitem__(self, index):
        id = self.ids[index]
        image = self._load_image(id)
        ann = self._load_target(id)

        return {
            "image": self.vis_processor(image),
            "text_input": self.text_processor(ann[random.randrange(len(ann))]["caption"]),
            "image_id": id
        }


class COCOEvalDataset(COCOBaseDataset):
    def __init__(self, img_dir, ann_file, vis_processor, text_processor):
        super(COCOEvalDataset, self).__init__(img_dir, ann_file, vis_processor, text_processor)
        
    # evaluation은 어떻게 진행되나?
    # 캡션이 5개인데 평가 방법은?
    def __getitem__(self, index):
        id = self.ids[index]
        image = self._load_image(id)
        ann = self._load_target(id)

        return {
            "image": self.vis_processor(image),
            "text_input": self.text_processor(ann[random.randrange(len(ann))]["caption"]),
            "image_id": id
        }