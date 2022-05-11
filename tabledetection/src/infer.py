import torch
import numpy as np
import json
import os
import copy
import itertools
import os
import random
import torch
import detectron2
import pandas as pd
from detectron2.data import detection_utils as utils
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger
setup_logger()

import random
import cv2
import matplotlib.pyplot as plt
from matplotlib import patches
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.cuda.empty_cache()


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


def register_dataset(df, dataset_label, image_dir):

    # Register dataset
    DatasetCatalog.register(
        dataset_label, lambda d=df: custom_dataset(df, image_dir))
    MetadataCatalog.get(dataset_label).set(thing_classes=["Table"])
    return MetadataCatalog.get(dataset_label), dataset_label

    
def main():

    print(torch.__version__, " , ", torch.cuda.is_available())

    VAL_DIR = 'data/val2021'
    TEST_DIR = 'data/test2021'
    test_df = pd.read_csv('data/annotations/diplast2021_test.csv')
    _, test_dataset = register_dataset(
        test_df, dataset_label='test_dataset', image_dir=TEST_DIR)

    UOS_MODEL_USE = 1

    if UOS_MODEL_USE == 1:
        UOS_MODEL = 'diplastmodel/faster_rcnn_R_101_FPN_3x_config.yaml'
        UOS_WEIGHT_PATH = 'diplastmodel/model_final.pth'
        
    elif UOS_MODEL_USE == 2:
        UOS_MODEL = ''
        UOS_WEIGHT_PATH = ''
    
    DatasetCatalog.clear()
    
    # Dataset for testing and evaluation
    register_coco_instances("val_coco", {}, "data/annotations/instances_val2021.json", "data/val2021/images")
    register_coco_instances("test_coco", {}, "data/annotations/instances_test2021.json", "data/test2021/images")
    
    # Testing and Evaluation Setup
    cfg = get_cfg()
    cfg.merge_from_file(UOS_MODEL)
    #cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.WEIGHTS = UOS_WEIGHT_PATH
    cfg.DATASETS.TEST = (test_dataset,)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75   # set the testing threshold for this model
    
    cfg.OUTPUT_DIR = "diplastmodel"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
                            
    #Inference
    print("Inference started !", flush=True)

    mde_test_metadata = MetadataCatalog.get("test_coco")
    predictor = DefaultPredictor(cfg)
    img_path = 'data/test2021/images/114_0.jpg'
    im = cv2.imread(img_path)
    outputs = predictor(im)
    print(outputs["instances"].pred_classes)
    print(outputs["instances"].pred_boxes)
    #v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TEST[0]))
    v = Visualizer(im[:, :, ::-1], metadata=mde_test_metadata, scale=5)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    
    imgfilename = os.path.splitext(os.path.basename(img_path))[0]
    pred_imgfile = str('inferimg/' + imgfilename+'.jpg')
   
    cv2.imwrite(pred_imgfile, v.get_image()[:, :, ::-1])
    
    print("Inference completed !", flush=True)
    
                            
if __name__ == "__main__":
    print("Program starts !", flush=True)
    main()
    print("Program ends !", flush=True)
