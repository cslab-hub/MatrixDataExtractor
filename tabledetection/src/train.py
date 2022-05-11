import numpy as np
import json
import os
import copy
import itertools
import os
import random
import yaml
import pandas as pd
import cv2
import torch
import detectron2

from detectron2.config import get_cfg
from detectron2.structures import BoxMode
from detectron2.data import detection_utils as utils
import detectron2.data.transforms as T
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.logger import setup_logger
setup_logger()

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.cuda.empty_cache()

'''

Prepare Dataset: 

'''

def custom_dataset(df, dir_image):

    dataset_dicts = []

    for image_id, image_name in enumerate(df.image_id.unique()):

        record = {}
        image_df = df[df['image_id'] == image_name]
        file_name = image_df['file_name'].astype(str).unique()
        file_name = file_name[0]
        img_path = os.path.join(dir_image, 'images/' + file_name)

        record['file_name'] = img_path
        record['image_id'] = image_df['image_id']
        record['height'] = int(image_df['height'].values[0])
        record['width'] = int(image_df['width'].values[0])

        objs = []
        for _, row in image_df.iterrows():

            x_min = int(row.xmin)
            y_min = int(row.ymin)
            x_max = int(row.xmax)
            y_max = int(row.ymax)
            label = str(row.label)

            if (label.lower() == 'table'):
                category_id = 0
            
            obj = {
                "bbox": [x_min, y_min, x_max, y_max],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": category_id,
                "iscrowd": 0

            }

            objs.append(obj)

        record['annotations'] = objs
        dataset_dicts.append(record)

    return dataset_dicts


'''
Custom Mapper: Data Augmentation
'''
def custom_mapper(dataset_dict):
    # it will be modified by code below
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict["file_name"], format="BGR")

    transform_list = [T.Resize((1333, 800)),
                      T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
                      T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
                      ]
    image, transforms = T.apply_transform_gens(transform_list, image)
    dataset_dict["image"] = torch.as_tensor(
        image.transpose(2, 0, 1).astype("float32"))

    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    return dataset_dict


class DiPlastTrainer(DefaultTrainer):

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper)
    

def register_dataset(df, dataset_label, image_dir):

    # Register dataset
    DatasetCatalog.register(
        dataset_label, lambda d=df: custom_dataset(df, image_dir))
    MetadataCatalog.get(dataset_label).set(thing_classes=["Table"])
    return MetadataCatalog.get(dataset_label), dataset_label

# Save our config file
def save_config_yaml(cfg):
    dict_ = yaml.safe_load(cfg.dump())
    with open(os.path.join(cfg.OUTPUT_DIR, 'uos_dip_config.yaml'), 'w') as file:
        _ = yaml.dump(dict_, file)
        

# Training Setup
def cfg_setup(MODEL, WEIGHT_PATH, train_dataset, val_dataset):
    
    # Tablebank - LayoutParser
    cfg = get_cfg()
    cfg.merge_from_file(MODEL)
    cfg.MODEL.DEVICE = "cuda"
    cfg.MODEL.WEIGHTS = WEIGHT_PATH
    cfg.DATASETS.TRAIN = (train_dataset,)
    cfg.DATASETS.TEST = (val_dataset,)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.MASK_ON = False
    cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675]
    cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0]
    # https://detectron2.readthedocs.io/en/latest/modules/modeling.html
    # https://detectron2.readthedocs.io/en/latest/modules/modeling.html#detectron2.modeling.ResNet.freeze
    cfg.MODEL.BACKBONE.FREEZE_AT = 2

    # Lower to reduce memory usage (1 is the lowest)
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR =  0.00025 # Base learning rate
    cfg.SOLVER.MAX_ITER = 50000  # Maximum number of iterations
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    
    return cfg

    
def main():

    print(torch.__version__, " , ", torch.cuda.is_available())

    # Mistakes: filepath carefully inserted
    TRAIN_DIR = 'data/train2021'
    VAL_DIR = 'data/val2021'
    TEST_DIR = 'data/test2021'

    MODEL_USE = 2

    if MODEL_USE == 1:
        MODEL = 'model/configs/mask_rcnn_R_50_FPN_3x_config.yml'
        WEIGHT_PATH = 'model/configs/mask_rcnn_R_50_FPN_3x_model_final.pth'
    elif MODEL_USE == 2:
        MODEL = 'model/configs/faster_rcnn_R_101_FPN_3x_config.yaml'
        WEIGHT_PATH = 'model/configs/faster_rcnn_R_101_FPN_3x_model_final.pth'

    
    train_df = pd.read_csv('data/annotations/diplast2021_train.csv')
    val_df = pd.read_csv('data/annotations/diplast2021_val.csv')
    test_df = pd.read_csv('data/annotations/diplast2021_test.csv')
    print("Train df: ", len(train_df), ", Val df: ", len(val_df), ", Test df: ", len(test_df))
    
    DatasetCatalog.clear()

    metadata, train_dataset = register_dataset(
        train_df, dataset_label='train_diplast', image_dir=TRAIN_DIR)
    _, val_dataset = register_dataset(
        val_df, dataset_label='val_diplast', image_dir=VAL_DIR)
    
    print("train_metadata: ", metadata.thing_classes)
    
    cfg = cfg_setup(MODEL, WEIGHT_PATH, train_dataset, val_dataset)
    
    cfg.OUTPUT_DIR = "diplastmodel"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    save_config_yaml(cfg)
                            
    #Training
    print("Training started !", flush=True)
    trainer = DiPlastTrainer(cfg)
    
    # Use to visualize training data
    '''
    train_data_loader = trainer.build_train_loader(cfg)
    train_data_iter = iter(train_data_loader)
    batch = next(train_data_iter)
    '''                        
    trainer.resume_or_load(resume=False)
    trainer.train()
                            
    # Use to visualize Model Architecture
    '''
    model = build_model(cfg)
    print('Model', model)
    '''
    print("Training completed !", flush=True)
    
                            
if __name__ == "__main__":
    print("Program starts !", flush=True)
    main()
    print("Program ends !", flush=True)
