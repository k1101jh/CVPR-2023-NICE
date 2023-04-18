import os
import sys
import nltk
import argparse
import logging
import datetime
import json
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.transforms import Lambda
from torch.utils.tensorboard import SummaryWriter

from dataset_config import COCO_dataset_config
from codes.train_configuration import DatasetType, TrainConfiguration
from custom_datasets.coco_dataset import COCODataset
from models.my_model import SimpleEncoder, SimpleDecoder
from utils import show_image_caption



def train(model, dataloader):
    logging.info("Start Training...")
    for image, caption in tqdm(dataloader):
        # plt.imshow(image[0])
        # print(caption[0])
        show_image_caption(image, caption, num_images=9, save_path='./sample.png')
        break
        
        
if __name__ == '__main__':
    # parsing
    parser = argparse.ArgumentParser(description='Train Model')
    parser.add_argument('--dataset_type', type=DatasetType, choices=list(DatasetType), default=DatasetType.COCO, required=False, help='Dataset to use')
    parser.add_argument('--lr', type=float, default=1e-4, required=False, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=1, required=False, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, required=False, help='Num epochs')
    parser.add_argument('--embed_size', type=int, default=256, required=False, help='Embedding vector size')
    parser.add_argument('--load_config', type=str, default=None, required=False, help='Load config')
    args = parser.parse_args()
    
    # train result save dir name
    results_dir = 'results'
    result_dirname = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    result_dir_fullpath = os.path.join(results_dir, result_dirname)
    os.makedirs(result_dir_fullpath, exist_ok=True)
    
    # logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s: %(message)s",
        level=logging.INFO,
        datefmt="%I:%M:%S",
        handlers=[
            logging.FileHandler(os.path.join(result_dir_fullpath, 'train.log')),
            logging.StreamHandler(sys.stdout),
        ])
    
    # tensorboard
    writer = SummaryWriter(os.path.join('runs', result_dirname))
    config_str = json.dumps(vars(args))
    writer.add_text('configs', config_str)

    # config
    config_save_path = os.path.join(result_dir_fullpath, 'config.json')
    
    if args.load_config is not None:
        config = TrainConfiguration.load_config(args.load_config)
    else:
        config = TrainConfiguration(**vars(args))
        
        # config = Configuration(
        #     dataset_type=DatasetType.COCO,
        #     epochs=100,
        #     lr=1e-4,
        #     batch_size=32,
        #     embed_size=args.embed_size
        # )
    
        config.save_config(config_save_path)
    config.print_configs()
        
    # set device
    device = 'cuda:0'
    print('Use ', torch.cuda.get_device_name(device))
    
    # train ------------------------------------------------------
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])
    
    # __getitem__ 호출마다 target_transform 실행
    dataset = dset.CocoCaptions(
        root=COCO_dataset_config['train_folder'],
        annFile=COCO_dataset_config['train_caption'],
        transform=train_transform,
        target_transform=Lambda(lambda x: nltk.tokenize.word_tokenize(str(x).lower()))
    )
    
    # dataset 생성 시 모든 caption에 대해 target_transform 실행 후 저장
    # dataset = COCO_Dataset(
    #     img_dir=COCO_dataset_config['train_folder'],
    #     ann_file=COCO_dataset_config['train_caption'],
    #     transforms=train_transform,
    #     tokenize_func=nltk.tokenize.word_tokenize
    # )

    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    
    # # model 정의
    # encoder = SimpleEncoder(config.embed_size)
    # encoder.to(device)
    
    # decoder = SimpleDecoder(config.embed_size, hidden_size=100, vocab_size=len(dataloader.dataset.vocab), num_layers=1)
    # decoder.to(device)
    
    model = None
    
    train(model, dataloader)