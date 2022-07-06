import xml.etree.ElementTree as ET
import pathlib
from pathlib import Path
import torch
import numpy as np
import json
import os
import copy
import itertools
import os
import random
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

'''
def crop_save_img(infer_tableimg_dir, img, table_img_filename, table_no, x_min, y_min, x_max, y_max):
    Path(infer_tableimg_dir).mkdir(parents=True, exist_ok=True)
    table_img = infer_tableimg_dir + "/" + str(table_img_filename) + "_" + str(table_no) + ".jpg"
    imgCrop = img[y_min: y_max, x_min:x_max]
    cv2.imwrite(table_img, imgCrop)
'''

def get_predictor():

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
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.WEIGHTS = UOS_WEIGHT_PATH
    cfg.DATASETS.TEST = (test_dataset,)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75   # set the testing threshold for this model
    
    cfg.OUTPUT_DIR = "diplastmodel"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
                            
    #Inference
    #print("Inference started !", flush=True)

    mde_test_metadata = MetadataCatalog.get("test_coco")
    predictor = DefaultPredictor(cfg)
    return predictor, mde_test_metadata

def infer_table_from_all_doc(predictor, mde_test_metadata):
    prop_filename = 'util/prop/coac_prop.xml'
    root = ET.parse(prop_filename).getroot()
    for child_element in root.findall('dataset/tabledetection'):
        temp_imgdir = os.getcwd()  + child_element.find('tempimgdir').text
        infer_dir = os.getcwd() + child_element.find('inferimgdir').text
        infer_tableimg_dir = os.getcwd() + child_element.find('infertableimgdir').text

    for tdsname in os.listdir(temp_imgdir):
        tdsdir = os.path.join(temp_imgdir,tdsname)
        infer_tdsdir = os.path.join(infer_dir,tdsname)
        Path(infer_tdsdir).mkdir(parents=True, exist_ok=True)

        for filename in os.listdir(tdsdir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                img_path = os.path.join(tdsdir,filename)

                if os.path.isfile(img_path):
                    img = cv2.imread(img_path)
                    outputs = predictor(img)
                    print(outputs["instances"].pred_classes)
                    print(outputs["instances"].pred_boxes)
                    #v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TEST[0]))
                    v = Visualizer(img[:, :, ::-1], metadata=mde_test_metadata, scale=1) # Keep scale=1 for saving predicted images
                    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                    output_path = infer_tdsdir + "/" + filename
                    inferred_img = v.get_image()[:, :, ::-1]
                    cv2.imwrite(output_path, inferred_img)

                    table_img_filename = os.path.splitext(os.path.basename(filename))[0]

                    '''------ Extract Only Table section ------'''
                    '''----------Extract Bounding Box----------'''
                    bboxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()

                    for table_no, item in enumerate(bboxes):
                        if item.size == 4:
                            x_min = int(item[0]) # First co-ordinates
                            y_min = int(item[1])
                            x_max = int(item[2]) # Fourth co-ordinates
                            y_max = int(item[3])
                            
                            #print(idx, " : " , x_min, "," ,y_min, ",", x_max, ",", y_max)
                            index_name = str(table_img_filename) + "_" + str(table_no)
                            temp_df = pd.DataFrame({
                                "Filename": table_img_filename,
                                "TABLE_NO": table_no,
                                "X_MIN": x_min,
                                "Y_MIN": y_min,
                                "X_MAX": x_max,
                                "Y_MAX": y_max
                            }, index=[index_name])

                            column_names = ["Filename", "TABLE_NO", "X_MIN", "Y_MIN", "X_MAX", "Y_MAX"]
                            df = pd.DataFrame(columns=column_names)
                            df = pd.concat([df, temp_df])
                            # Save BBox information in CSV file
                            csv_filename = 'util/bbox/' + tdsname + '.csv'
                            df.to_csv(csv_filename,header=False,sep='\t',mode='a',index=False, encoding='utf-8', na_rep='Unkown')
                            #crop_save_img(infer_tableimg_dir, img, table_img_filename, table_no, x_min, y_min, x_max, y_max)

                    
                else:
                    print("Images not found !")
    
    #print("Inference completed !", flush=True)
    
def infer_table_from_single_doc(predictor,mde_test_metadata, tdsname):
    prop_filename = 'util/prop/coac_prop.xml'
    root = ET.parse(prop_filename).getroot()
    for child_element in root.findall('dataset/tabledetection'):
        temp_imgdir = os.getcwd()  + child_element.find('tempimgdir').text
        infer_dir = os.getcwd() + child_element.find('inferimgdir').text
        #infer_tableimg_dir = os.getcwd() + child_element.find('infertableimgdir').text

    tdsdir = os.path.join(temp_imgdir,tdsname)
    infer_tdsdir = os.path.join(infer_dir,tdsname)

    if os.path.isdir(tdsdir):
        Path(infer_tdsdir).mkdir(parents=True, exist_ok=True)
        for filename in os.listdir(tdsdir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                img_path = os.path.join(tdsdir,filename)

                if os.path.isfile(img_path):
                    img = cv2.imread(img_path)
                    outputs = predictor(img)
                    print(outputs["instances"].pred_classes)
                    print(outputs["instances"].pred_boxes)
                    #v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TEST[0]))
                    v = Visualizer(img[:, :, ::-1], metadata=mde_test_metadata, scale=1) # Keep scale=1 for saving predicted images
                    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                    output_path = infer_tdsdir + "/" + filename
                    inferred_img = v.get_image()[:, :, ::-1]
                    cv2.imwrite(output_path, inferred_img)

                    table_img_filename = os.path.splitext(os.path.basename(filename))[0]

                    '''------ Extract Only Table section ------'''
                    '''----------Extract Bounding Box----------'''
                    bboxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()

                    for table_no, item in enumerate(bboxes):
                        if item.size == 4:
                            x_min = int(item[0]) # First co-ordinates
                            y_min = int(item[1])
                            x_max = int(item[2]) # Fourth co-ordinates
                            y_max = int(item[3])
                        
                            #print(idx, " : " , x_min, "," ,y_min, ",", x_max, ",", y_max)
                            index_name = str(table_img_filename) + "_" + str(table_no)
                            temp_df = pd.DataFrame({
                                "Filename": table_img_filename,
                                "TABLE_NO": table_no,
                                "X_MIN": x_min,
                                "Y_MIN": y_min,
                                "X_MAX": x_max,
                                "Y_MAX": y_max
                            }, index=[index_name])

                            column_names = ["Filename", "TABLE_NO", "X_MIN", "Y_MIN", "X_MAX", "Y_MAX"]
                            df = pd.DataFrame(columns=column_names)
                            df = pd.concat([df, temp_df])
                            # Save BBox information in CSV file
                            csv_filename = 'util/bbox/' + tdsname + '.csv'
                            df.to_csv(csv_filename,header=False,sep='\t',mode='a',index=False, encoding='utf-8', na_rep='Unkown')
                            #crop_save_img(infer_tableimg_dir, img, table_img_filename, table_no, x_min, y_min, x_max, y_max)

                        
                else:
                    print("Images not found !")

    else:
        print("Directory does not exist. Please check your directory name.")   

    #print("Inference completed !", flush=True)


def main():
    predictor, mde_test_metadata = get_predictor()
    ''' Infer all PDF document images '''
    #infer_table_from_all_doc(predictor,mde_test_metadata)
    ''' Infers only single PDF document image '''
    tdsname = 'Your folder name from coac/util/data/tempimgdir directory'
    infer_table_from_single_doc(predictor,mde_test_metadata, tdsname)

if __name__ == "__main__":
    print("Program starts !", flush=True)
    main()
    print("Program ends !", flush=True)
