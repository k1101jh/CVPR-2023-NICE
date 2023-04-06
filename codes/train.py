from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.coco_dataset import COCO_Dataset
from dataset_config import COCO_dataset_config
from configuration import DatasetType, Configuration


def train(model, dataloader):
    for image, caption in tqdm(dataloader):
        print(image, caption)
        
        
if __name__ == '__main__':
    config_save_path = 'configs/sample_config.json'
    load_config = False
    
    if load_config:
        config = Configuration.load_config(config_save_path)
    else:
        config = Configuration(
            dataset_type=DatasetType.COCO,
            epochs=100,
            learning_rate=1e-4,
            batch_size=32,
        )
    
        config.save_config(config_save_path)
    
    
    model = None
    dataset = COCO_Dataset(
        image_folder=COCO_dataset_config['train_folder'],
        annotation_file=COCO_dataset_config['train_caption'],
        tokenizer=None,
        transform=None)
    dataloader = DataLoader(dataset, )
    
    train(model, dataset)