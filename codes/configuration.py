import os
import sys
import json
from enum import IntEnum, auto
from utils import get_YN_answer


class DatasetType(IntEnum):
    COCO = auto()
    CC3M = auto()
    CC12M = auto()
    LAION400M = auto()
    LAION5B = auto()
    YFCC15M = auto()
    YFCC100M = auto()


class Configuration:
    def __init__(self, dataset_type, epochs, learning_rate, batch_size):
        self.dataset_type = dataset_type
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
    def save_config(self, file_path):
        os.makedirs(os.path.dirname, exist_ok=True)
        if(os.path.exists(file_path)):
            if not get_YN_answer("같은 config 파일이 존재합니다. 계속 진행합니까?"):
                print("프로그램을 종료합니다.")
                sys.exit(-1)
        else:
            with open(file_path, "w") as json_file:
                json.dump(self.__dict__, json_file, sort_keys=True, indent=4)
    
    @classmethod
    def load_config(cls, file_path):
        # 파일 로드하고 configuration 생성하고 반환
        assert(os.path.exists(file_path))
        with open(file_path, 'r') as json_file:
            json_obj = json.load(json_file)
            return Configuration(**json_obj)