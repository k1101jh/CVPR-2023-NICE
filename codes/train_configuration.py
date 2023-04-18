import os
import sys
import json
from enum import Enum, IntEnum, auto
from utils import get_YN_answer


class DatasetType(IntEnum):
    COCO = auto()
    CC3M = auto()
    CC12M = auto()
    LAION400M = auto()
    LAION5B = auto()
    YFCC15M = auto()
    YFCC100M = auto()


class TrainConfiguration:
    def __init__(self, **kwargs):
        self.dataset_type = kwargs.pop("dataset_type", None)
        self.epochs = kwargs.pop("epochs", 100)
        self.lr = kwargs.pop("lr", 1e-4)
        self.batch_size = kwargs.pop("batch_size", 1)
        self.embed_size = kwargs.pop("embed_size", 256)
    
    def print_configs(self):
        members = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        print("Configs:")
        for member in members:
            value = self.__getattribute__(member)
            if isinstance(value, DatasetType):
                print(f"\t{member}: {DatasetType(value).name}")
            else:
                print(f"\t{member}: {value}")
        print('-' * 20)
        
    def save_config(self, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if(os.path.exists(file_path)):
            if not get_YN_answer("같은 config 파일이 존재합니다. 계속 진행합니까?", default="Y"):
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
            return cls(**json_obj)
