# Copyright (c) ArnabGhoshChowdhury, Universität Osnabrück and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from django.conf import settings
import pathlib, re, os
from pathlib import Path
from numpy.core.fromnumeric import product
import pandas as pd
from pdf2image import convert_from_path
import pytesseract
from pytesseract import Output
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import imutils, csv, json, random
from pymongo import MongoClient
import camelot

'''--------------------PyTorch Library --------------------'''
import torch
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.modeling import build_model
from detectron2.structures import BoxMode
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
'''--------------------Matrix Text Extractor--------------------'''
torch.manual_seed(42)
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


class MatrixTextExtractor(object):

    def __init__(self, prop_filename):

        '''
        manufacturer = "LyondellBasell"
        product = "Circulen 2420D Plus"
        filename = "Circulen 2420D Plus.pdf"
        #img_type_no = 1
        start_str = "Product Description"
        end_str = "Regulatory Status"
        lookup_text_list = ["Page 1 of 3", "Page 2 of 3", "Page 3 of 3" ]
        '''

        '''----------Extract data from XML Configuration File----------'''
        root = ET.parse(prop_filename).getroot()
        for child_element in root.findall('dataset/tabledetection'):
            '''
            Current Directory + Path of other dirs retrieved from XML files
            '''
            self.base_pdf_dirname = os.getcwd()  + child_element.find('sourcepdf').text
            self.parent_temptextimg_dirname = os.getcwd()  + child_element.find('temptextimage').text
            self.parent_temptableimg_dirname = os.getcwd()  + child_element.find('temptableimage').text
            self.temp_imgtotext_dir = os.getcwd()  + child_element.find('imgtotext').text
            self.temp_tabulardata_dir = os.getcwd()  + child_element.find('tabulardata').text
            self.infertableimg = os.getcwd() + child_element.find('infertableimg').text
            self.storetableimg = os.getcwd() + child_element.find('storetableimg').text


    '''
    Apply OCR to extract text from a set of images.

    Input: A set of image files.
    Output: Return full text extracted from PDF. Save corrsponding text file in MongoDB by calling post method.
    '''
    def get_full_text_per_image(self, img, temp_imgtotext_filename):
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #retval,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        '''
        img_type_no = int(self.img_type_no)
        if img_type_no == 1:
        '''
        custom_config = r'-c preserve_interword_spaces=1 --oem 1 --psm 3 -l eng'
        # https://stackoverflow.com/questions/59582008/preserving-indentation-with-tesseract-ocr-4-x
        d = pytesseract.image_to_data(img, config=custom_config, output_type=Output.DICT)
        df = pd.DataFrame(d)
        # clean up blanks
        df1 = df[(df.conf!='-1')&(df.text!=' ')&(df.text!='')]
        # sort blocks vertically
        sorted_blocks = df1.groupby('block_num').first().sort_values('top').index.tolist()
        for block in sorted_blocks:
            curr = df1[df1['block_num']==block]
            sel = curr[curr.text.str.len()>3]
            char_w = (sel.width/sel.text.str.len()).mean()
            prev_par, prev_line, prev_left = 0, 0, 0
            text = ''
            for ix, ln in curr.iterrows():
                # add new line when necessary
                if prev_par != ln['par_num']:
                    text += '\n'
                    prev_par = ln['par_num']
                    prev_line = ln['line_num']
                    prev_left = 0
                elif prev_line != ln['line_num']:
                    text += '\n'
                    prev_line = ln['line_num']
                    prev_left = 0

                added = 0  # num of spaces that should be added
                if ln['left']/char_w > prev_left + 1:
                    added = int((ln['left'])/char_w) - prev_left
                    text += ' ' * added 
                text += ln['text'] + ' '
                prev_left += len(ln['text']) + added + 1
            text += '\n'
            with open(temp_imgtotext_filename, 'a', encoding= 'utf-8') as text_file:
                text_file.write(text)
                text_file.write("\n") 
        

    def diplast_ocr(self, temp_imgfile_dir, manufacturer, tds, imagefile_set):
        txt_manufacturer_dir = self.temp_imgtotext_dir + "/" + manufacturer
        temp_imgtotext_filename =  txt_manufacturer_dir + "/" + str(tds).rstrip(".pdf") + "_preprocessed.txt"
        manufacturer_dir = pathlib.Path(txt_manufacturer_dir)
        manufacturer_dir.mkdir(parents=True, exist_ok=True)
        print("manufacturer_dir: ", manufacturer_dir)
        #Create empty file
        open(temp_imgtotext_filename, 'w+').close()

        for imagefile in imagefile_set:
            img = cv2.imread(imagefile, 0)
            self.get_full_text_per_image(img, temp_imgtotext_filename)
            '''
            xml = pytesseract.image_to_alto_xml(imagefile)
            xml_filename = imagefile + '.xml'
            with open(xml_filename, 'w+b') as xml_file:
                xml_file.write(xml) 
            xml_file.close()
            '''
        imagefile_set.clear()
        return temp_imgtotext_filename
    
    '''
    Read property file, take manufacturer name, TDS name, table image names.
    Merge as manf_tds_imgid.jpg format

    Input: property filename and plastic product name retrieved from plastic product technical data sheet.
    Output: Return None. Stores images and text files in corresponding directory.
    
    '''
    def merge_table_img(self, manufacturer, tds) :
        tds = str(tds).strip('\'')
        print("manufacturer:", manufacturer)
        ''' Check Parent image_dir (util/data/tempimg/textdata) and create subdir(Manufacturer Names) if not exists '''
        Path(self.storetableimg).mkdir(parents=True, exist_ok=True)
        Path(self.storetableimg + "/" + manufacturer).mkdir(parents=True, exist_ok=True)

        ''' Get 'infertableimg' directory information '''
        tds = tds.replace('.pdf','').replace('\'','')
        dir_image = self.infertableimg + "/" + manufacturer + "/" + tds
        target_img_dir = self.storetableimg + "/" + manufacturer
        
        imagelist = Path(dir_image).rglob('*.jpg')
        imagefile_set = set()
        imagefile_set.clear()
        for imagename in imagelist:
            image_in_str = str(imagename).strip("")
            imagefile_set.add(image_in_str)

        imagefile_set = sorted(imagefile_set)
       
        ''' Save image files in single folder'''
        table_img_set = set()
        for img_path in imagefile_set:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            table_img_file = os.path.basename(img_path)
            table_img_set.add(str(table_img_file))
            print("table_img_file: ", table_img_file)
            imagefile = target_img_dir + "/" + tds + "_" + table_img_file
            '''Save images'''
            cv2.imwrite(imagefile,img)

        return table_img_set




    '''
    Read property file, split single column layout PDF and convert to images.
    Pre-process images and extract text using OCR.

    Input: property filename and plastic product name retrieved from plastic product technical data sheet.
    Output: Return None. Stores images and text files in corresponding directory.
    '''
    def pdf_to_text(self, manufacturer, tds) :
        tds = str(tds).strip('\'')
        print("manufacturer:", manufacturer)
        ''' Check Parent image_dir (util/data/tempimg/textdata) and create subdir(Manufacturer Names) if not exists '''
        Path(self.parent_temptextimg_dirname).mkdir(parents=True, exist_ok=True)
        Path(self.parent_temptextimg_dirname + "/" + manufacturer).mkdir(parents=True, exist_ok=True)

        ''' Source TDS PDF file uploaded by User '''
        search_filepath= self.base_pdf_dirname+ "/" + manufacturer + "/" + tds #productname + ".pdf"
        print("search_filepath: ", search_filepath)
        ''' Create temporary img dir by considering each TDS PDF filename(without .pdf) '''
        temp_imgfile_dirpath = self.parent_temptextimg_dirname + "/" + manufacturer + "/" + str(tds).rstrip(".pdf") #productname
        print("temp_imgfile_dirpath: ", temp_imgfile_dirpath)
        
        pdf_dir = Path(self.base_pdf_dirname)
        temp_img_dir = pathlib.Path(self.parent_temptextimg_dirname)
        temp_imgfile_dir = pathlib.Path(temp_imgfile_dirpath)

        if (pdf_dir.exists()):
            search_file = Path(search_filepath)
            if search_file.exists():
                pass
            else:
                print("File not found in source dir !")
        else:
            print("Source PDF directory does not exist !")
        
        ''' Split PDF and convert into images '''

        if(temp_img_dir.exists()):
            temp_imgfile_dir.mkdir(parents=True, exist_ok=True)
            pages = convert_from_path(search_file, dpi=72)
            pg_no = 0
            for page in pages:
                page.save( temp_imgfile_dirpath + "/" + str(pg_no)+ '.jpg', 'JPEG')
                pg_no = pg_no + 1
            
        else:
            print("Temporary image directory does not exist !")


        imagelist = Path(temp_imgfile_dirpath).rglob('*.jpg')
        imagefile_set = set()
        imagefile_set.clear()
        for imagename in imagelist:
            image_in_str = str(imagename)
            imagefile_set.add(image_in_str)
        imagefile_set = sorted(imagefile_set)

        #img_type_no = int(self.img_type_no)
        for imagefile in imagefile_set:
            product_dir = self.temp_imgtotext_dir + str(tds).strip(".pdf")
            img = cv2.imread(imagefile)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Remove logo
            cv2.imwrite(imagefile,img)
            print("Image write complete !")
            
        temp_imgtotext_filename = self.diplast_ocr(temp_imgfile_dir, manufacturer, tds, imagefile_set)
        return temp_imgtotext_filename
        
        '''
        try:
            shutil.rmtree(temp_imgfile_dir)
            shutil.rmtree(product_dir)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))
        '''
     

    def text_preprocessing(self):            
        
        #text_filelist = glob.glob(self.temp_imgtotext_dir + "*_preprocessed.txt")
        text_filelist = self.temp_imgtotext_dir + self.filename.strip(".pdf") + "_preprocessed.txt"
        ignored_line_list = list()
        #img_type_no = self.img_type

        for text_file in text_filelist:
            with open(text_file, 'r') as f:
                for lineno, line in enumerate(f, 1):
                    for lookup_text in self.lookup_text_list:
                        if lookup_text.strip() in line:
                            ignored_line_list.append(lineno)
                            '''
                            if img_type_no == 1:
                                ignored_line_list.append(lineno)
                                ignored_line_list.append(lineno-1)
                                ignored_line_list.append(lineno-2)

                            if img_type_no == 2:
                                ignored_line_list.append(lineno)
                            '''
              
            f_in = text_file
            f_out = text_file.replace('_preprocessed', '_postprocessed')
            # https://stackoverflow.com/questions/36208725/python-delete-a-specific-line-number
            with open(f_in, 'r') as fin, open(f_out, 'w') as fout:
                for lineno, line in enumerate(fin, 1):
                    if lineno not in ignored_line_list:
                        if not line.strip(): 
                            continue
                        #print(line)
                        fout.write(line)
                        
        '''
        del_filelist = glob.glob(temp_imgtotext_dir + "*_preprocessed.txt")
        for filename in del_filelist:
            try:
                os.remove(filename)
            except OSError as e:
                print("Error: %s : %s" % (f, e.strerror))
        '''


    def extract_text_by_key(self, productname, firsttext, secondtext):

        #text_filelist = glob.glob(self.temp_imgtotext_dir + "*_postprocessed.txt")
        #text_filelist = glob.glob(self.temp_imgtotext_dir + "*_preprocessed.txt")
        text_file = self.temp_imgtotext_dir + productname + "_preprocessed.txt"
        
        '''
        Last line of text file and last word also.
        '''

        #for text_file in text_filelist:
        with open(text_file, 'r') as f:
            file_content = f.read()

            if secondtext == '':
                line_list = file_content.splitlines()
                last_line = line_list[-1]
                last_line_words = last_line.split()
                last_line_word = last_line_words[-1]

            else:
                secondtext = secondtext
                
            if self.end_str == '':
                sub_str_list = re.findall('(?s)(?<={})(.*?)(?={})'.format(firsttext, last_line_word), file_content, flags=re.S)
                sub_str_list.append(last_line_word)
                    
            else:
                sub_str_list = re.findall('(?s)(?<={})(.*?)(?={})'.format(firsttext, secondtext), file_content, flags=re.S)
                
            sub_str = ''.join(str(text) for text in sub_str_list)

            # https://stackoverflow.com/questions/9347419/python-strip-with-n
            searchkey = str(firsttext)
            searchvalue = sub_str.strip('\n').strip('\t').replace('\n','').replace('\t','')
            return searchkey, searchvalue   


    '''--------------------Matrix Table Region Extractor (UI Interaction)--------------------'''
    '''
    Model inference on Technical Datasheets

    Input: property filename and plastic product name retrieved from plastic product technical data sheet.
    Output: Return None. Infer tables and stores in inferimg directory.
    '''
    def infer_table_on_tds(self, prop_filename, manufacturer, tds, scale=1, img_width=612, img_height=792):
        '''--- DPI Value in initialize to 72. Change DPI according to your requirements and adapt necessary code changes ---'''
        tds = str(tds).strip('\'')
        ''' Check Parent image_dir (util/data/tempimg/tabledata) and create subdir(Manufacturer Names) if not exists '''
        Path(self.parent_temptableimg_dirname).mkdir(parents=True, exist_ok=True)
        Path(self.parent_temptableimg_dirname + "/" + manufacturer).mkdir(parents=True, exist_ok=True)
       
        ''' Source TDS PDF file uploaded by User '''
        search_filepath= self.base_pdf_dirname+ "/" + manufacturer + "/" + tds #productname + ".pdf"
        print("search_filepath: ", search_filepath)
        ''' Create temporary img dir by considering each TDS PDF filename(without .pdf) '''
        temp_imgfile_dirpath = self.parent_temptableimg_dirname + "/" + manufacturer + "/" + str(tds).rstrip(".pdf") #productname
        print("temp_imgfile_dirpath: ", temp_imgfile_dirpath)
        
        pdf_dir = Path(self.base_pdf_dirname)
        temp_img_dir = pathlib.Path(self.parent_temptableimg_dirname)
        temp_imgfile_dir = pathlib.Path(temp_imgfile_dirpath)

        if (pdf_dir.exists()):
            search_file = Path(search_filepath)
            if search_file.exists():
                pass
            else:
                print("File not found in source dir !")
        else:
            print("Source PDF directory does not exist !")
        
        ''' Split PDF and convert into images '''

        if(temp_img_dir.exists()):
            temp_imgfile_dir.mkdir(parents=True, exist_ok=True)
            pages = convert_from_path(search_file, dpi=72)
            pg_no = 0
            for page in pages:
                page.save( temp_imgfile_dirpath + "/" + str(pg_no)+ '.jpg', 'JPEG')
                pg_no = pg_no + 1
            
        else:
            print("Temporary image directory does not exist !")



        '''-------------------- Transfer Learning Model Inference --------------------'''
        imagelist = Path(temp_imgfile_dirpath).rglob('*.jpg')
        imagefile_set = set()
        imagefile_set.clear()
        for imagename in imagelist:
            image_in_str = str(imagename)
            imagefile_set.add(image_in_str)

        imagefile_set = sorted(imagefile_set)
        mie_inferer = MatrixTableExtractor(prop_filename)

        '''-----Create Empty Dataframe-----'''
        df_columns = ['TDS', 'IMG_NO', 'TABLE_NO','X_MIN', 'Y_MIN', 'X_MAX', 'Y_MAX']
        df = pd.DataFrame(columns=df_columns)

        for img_path in imagefile_set:
            csv_filename, temp_df = mie_inferer.tableimg_model_infer(img_path, df, manufacturer, tds)
            df = pd.concat([df, temp_df])

        '''----------Sort based on img_no, y_min value and reset index----------'''
        df = df.sort_values(['IMG_NO','Y_MIN'], ascending=True)
        df = df.reset_index(drop=True)

        '''----------Write to CSV file----------'''
        df.to_csv(csv_filename, sep='\t', encoding='utf-8')

        '''----------Map image pixels to PDF co-ordinates----------'''
        df_table_list = pd.read_csv(csv_filename, sep='\t', index_col=None, encoding='utf-8')
        count_row = df_table_list.shape[0]  # Gives number of rows
        count_col = df_table_list.shape[1]  # Gives number of columns
        ''' Filter dataframe'''
        df_table_info_list = df_table_list[df_columns]
        #print(df_table_info_list)
        df_row = 0
        for df_row in range(count_row):
            pdf_pg_no = int (df_table_info_list.iloc[df_row]["IMG_NO"])
            table_no = int (df_table_info_list.iloc[df_row]["TABLE_NO"])
            table_x_min = df_table_info_list.iloc[df_row]["X_MIN"]
            table_x_max = df_table_info_list.iloc[df_row]["X_MAX"]
            table_y_min = df_table_info_list.iloc[df_row]["Y_MIN"]
            table_y_max = df_table_info_list.iloc[df_row]["Y_MAX"]
            #print("INFO-1: ", pdf_pg_no, table_x_min, table_x_max, table_y_min, table_y_max)
            '''
            Important Note: The PDF page size by default refers to US Letter, Portrait (216 x 279 mm) or (8.5 x 11 inches). It fits to our calculation.
            If you have different PDF page size width and height, then change code for scale, pdf_table_x_min, pdf_table_x_max, pdf_table_y_min, pdf_table_y_max.
            Detailed information is available on Wiki.
            '''
            pdf_table_x_min = table_x_min/scale
            pdf_table_x_max = table_x_max/scale

            pdf_table_y_min = int(img_height-table_y_min)/scale
            pdf_table_y_max = int(img_height-table_y_max)/scale

            table_excel_name = self.infertableimg+ "/" + manufacturer + "/" + str(tds).rstrip(".pdf") + "/" + str(pdf_pg_no) +"_" + str(table_no) + ".xlsx"
            #print("Info-2: ", pdf_table_x_min , pdf_table_y_min , pdf_table_x_max , pdf_table_y_max)
            pdf_table_coordinate = str(int(pdf_table_x_min)) +","+ str(int(pdf_table_y_min))+ ","+ str(int(pdf_table_x_max))+ ","+ str(int(pdf_table_y_max))
            #print("pdf_table_coordinate: ", pdf_table_coordinate)
            pdf_table = camelot.read_pdf(search_filepath, pages= str(1+pdf_pg_no), flavor='stream',  table_areas=[pdf_table_coordinate])
            df_table = pdf_table[0].df
            print(df_table)
            '''---Save dataframe to excel file---'''
            df_table.to_excel(table_excel_name)

        infer_status = "Table detected from your selected document(s) saved to \"util / data / tabledet / inference \". \
                        Please check \"inferimg\" and \"infertableimg\" folder for more details"
                        
        return infer_status
        
        '''
        try:
            shutil.rmtree(temp_imgfile_dir)
            shutil.rmtree(product_dir)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))
        '''
    '''-------------------- Get Cropped Table Region and Display on UI --------------------'''
    def get_table_imglist(self, manufacturer, tds, ocr_config='-c preserve_interword_spaces=1 --oem 3 --psm 6',  ocr_lang='eng'):
        tds = tds.replace('.pdf','').replace('\'','')
        dir_image = self.infertableimg + "/" + manufacturer + "/" + tds
        
        imagelist = Path(dir_image).rglob('*.jpg')
        imagefile_set = set()
        imagefile_set.clear()
        for imagename in imagelist:
            image_in_str = str(imagename).strip("")
            imagefile_set.add(image_in_str)

        imagefile_set = sorted(imagefile_set)
       
        ''' Save extracted file in .csv format using OCR'''
        table_img_set = set()

        for img_path in imagefile_set:
            image = cv2.imread(img_path)
            table_info = pytesseract.image_to_string(image, config=ocr_config,  lang=ocr_lang)
            table_info = table_info.replace('\x0c', '').replace('\\x0c', '').replace('?', '')
            table_df_file = os.path.splitext(os.path.basename(img_path))[0]

            '''Sending info to UI'''
            table_img_set.add(str(table_df_file) + ".jpg")
            '''Create corresponding CSV file against each JPG image'''
            table_txtfile = str(dir_image) + "/" + table_df_file + ".txt" #".csv"
            '''Write table information into CSV file'''
            with open(table_txtfile, 'w') as f:
                f.write(table_info)

        return table_img_set

    '''--------------------Import CSV to MongoDB --------------------'''
    # src: https://gist.github.com/jxub/f722e0856ed461bf711684b0960c8458
    def mongo_connect(self, db_name, coll_name):
        """ Imports a csv file at path csv_name to a mongo colection
        returns: count of the documants in the new collection
        """
        client = MongoClient("mongodb://localhost:27017/")
        db = client[db_name]
        coll = db[coll_name]
        '''--------------------Only Turn On below Swict if you need to clear all data--------------------'''
        #manufacturer_doc_count = coll.delete_many({})
        return db, coll
    
    def mongo_import(self, csv_path, db, coll, manufacturer, tds):
        '''
        data = pd.read_csv(csv_path)
        payload = json.loads(data.to_json(orient='records'))
        '''
        with open(csv_path, 'r') as f:
            tabular_data = f.read()

        csv_path = os.path.splitext(os.path.basename(csv_path))[0]
        payload = {
            "manufacturer": str(manufacturer), 
            "tds": str(tds),
            "csv_file": str(csv_path),
            "tabular_data" : tabular_data 
        }
        coll.insert(payload)

    '''--------------------Transfer CSV data to MongoDB database --------------------'''
    def transfer_csv_to_db(self, manufacturer, tds):
        tds = tds.replace('.pdf','').replace('\'','')
        dir_image = self.infertableimg + "/" + manufacturer + "/" + tds
        csv_list = Path(dir_image).rglob('*.txt')
        db, coll = self.mongo_connect("matrixtextapp", "matrixtextapp_tabledata")
        for csv_file in csv_list:
            self.mongo_import(csv_file, db, coll, manufacturer, tds)

        status_msg = "Tabular Data from selected datasheets are transffered to database"
        return status_msg


'''***************************************************************************************************************************************'''

'''--------------------MIE Table Detection Model Inference on Test dataset--------------------'''

class MatrixTableExtractor(object):
    def __init__(self, prop_filename):

        '''----------Extract data from XML Configuration File----------'''
        root = ET.parse(prop_filename).getroot()
        for child_element in root.findall('dataset/tabledetection'):
            '''
            Current Directory + Path of other dirs retrieved from XML files
            '''
            self.modelweight = os.getcwd() + child_element.find('modelweight').text
            self.configfile = os.getcwd() + child_element.find('configfile').text
            self.inferimg = os.getcwd() + child_element.find('inferimg').text
            self.infertableimg = os.getcwd() + child_element.find('infertableimg').text
            self.modeltestdir = os.getcwd() + child_element.find('modeltestdir').text
            self.modeltestdf = os.getcwd() + child_element.find('modeltestdf').text

    def custom_dataset(self, df, dir_image):

        dataset_dicts = list() #[]

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

            objs = list() #[]
            for _, row in image_df.iterrows():

                x_min = int(row.xmin)
                y_min = int(row.ymin)
                x_max = int(row.xmax)
                y_max = int(row.ymax)
                label = str(row.labels)

                if (label.lower() == 'table'):
                    category_id = 0
                """
                poly = [(x_min, y_min), (x_max, y_min),
                        (x_max, y_max), (x_min, y_max)]

                poly = list(itertools.chain.from_iterable(poly))
                """
                obj = {
                    "bbox": [x_min, y_min, x_max, y_max],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    #"segmentation": [poly],
                    "category_id": category_id,
                    "iscrowd": 0

                }

                objs.append(obj)

            record['annotations'] = objs
            dataset_dicts.append(record)

        return dataset_dicts

    def register_dataset(self, df, dataset_label, image_dir):
        DatasetCatalog.clear()
        # Register dataset
        DatasetCatalog.register(
            dataset_label, lambda d=df: self.custom_dataset(df, image_dir))
        MetadataCatalog.get(dataset_label).set(thing_classes=["Table"])
        return MetadataCatalog.get(dataset_label), dataset_label 

    def tableimg_model_infer(self, img_path, df, manufacturer, tds):
        img_no = str(os.path.basename(img_path).rstrip(".jpg"))
        
        '''-----Create Empty Dataframe-----'''
        df_columns = ['TDS', 'IMG_NO', 'TABLE_NO','X_MIN', 'Y_MIN', 'X_MAX', 'Y_MAX']
        df = pd.DataFrame(columns=df_columns)

        configfile = Path(self.configfile)
        modeltestdir = Path(self.modeltestdir)
        modeltestdf = Path(self.modeltestdf)
        tds = str(tds).strip(".pdf")
        Path(self.inferimg).mkdir(parents=True, exist_ok=True)
        Path(self.inferimg + "/" + manufacturer).mkdir(parents=True, exist_ok=True)
        Path(self.inferimg + "/" + manufacturer + "/" + tds).mkdir(parents=True, exist_ok=True)

       
        img_name = self.inferimg + "/" + manufacturer + "/" + tds + "/" + img_no + ".jpg"
        csv_filename = self.inferimg + "/" + manufacturer + "/" + tds + "/" + tds + ".csv"

        test_metadata, test_dataset = self.register_dataset(modeltestdf, dataset_label='test_diplast', image_dir=modeltestdir)

        cfg = get_cfg()
        cfg.merge_from_file(configfile)
        cfg.OUTPUT_DIR = self.modelweight
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        cfg.MODEL.DEVICE = "cpu"
        cfg.DATALOADER.NUM_WORKERS = 1
        cfg.SOLVER.IMS_PER_BATCH = 2
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        model = build_model(cfg)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75 
        predictor = DefaultPredictor(cfg)
        infer_img = cv2.imread(img_path)
        outputs = predictor(infer_img)
        v = Visualizer(infer_img[:, :, ::-1], metadata=test_metadata)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite(img_name, v.get_image()[:, :, ::-1])
        '''----------Extract Bounding Box----------'''
        bboxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()

        #bbox_count = 0
        for table_no, item in enumerate(bboxes):
            if item.size == 4:
                x_min = int(item[0]) # First co-ordinates
                y_min = int(item[1])
                x_max = int(item[2]) # Fourth co-ordinates
                y_max = int(item[3])
                
                #print(idx, " : " , x_min, "," ,y_min, ",", x_max, ",", y_max)
                index_name = str(img_no) + str(table_no)
                temp_df = pd.DataFrame({
                    "TDS": tds,
                    "IMG_NO": img_no,
                    "TABLE_NO": table_no,
                    "X_MIN": x_min,
                    "Y_MIN": y_min,
                    "X_MAX": x_max,
                    "Y_MAX": y_max
                }, index=[index_name])

                df = pd.concat([df, temp_df])
                self.crop_save_img(manufacturer, tds, infer_img, img_no, table_no, x_min, y_min, x_max, y_max)
                
                
            else:
                pass

        return csv_filename, df

    def crop_save_img(self, manufacturer, tds, infer_img, img_no, table_no, x_min, y_min, x_max, y_max):
        tds = str(tds).strip(".pdf")
        Path(self.infertableimg).mkdir(parents=True, exist_ok=True)
        Path(self.infertableimg + "/" + manufacturer).mkdir(parents=True, exist_ok=True)
        Path(self.infertableimg + "/" + manufacturer + "/" + tds).mkdir(parents=True, exist_ok=True)

        table_img = self.infertableimg + "/" + manufacturer + "/" + tds + "/" + str(img_no) + "_" + str(table_no) + ".jpg"
        imgCrop = infer_img[y_min: y_max, x_min:x_max]
        cv2.imwrite(table_img, imgCrop)

        

'''***************************************************************************************************************************************''' 

'''--------------------Matrix Information Extraction Manager--------------------'''   
                
'''***************************************************************************************************************************************'''


class MatrixInfoExtractorManager(object):    

    def __init__(self, prop_filename):
        self.prop_filename = prop_filename

    def merge_img_files_in_dir(self, manufacturer, tds):

            file_status = ''
            matrix_merge_tabimg = MatrixTextExtractor(self.prop_filename)
            
            #temp_imgtotext_filename =  matrix_extract_text.temp_imgtotext_dir + matrix_extract_text.filename.strip(".pdf") + "_preprocessed.txt"
            ''' Pre-processing by Opencv. Use of PyTessaract to extract text and save in database'''
            temp_imgtotext_filename = matrix_merge_tabimg.pdf_to_text(manufacturer, tds)
            print("CV temp_imgtotext_filename: ", temp_imgtotext_filename)
            temp_imgtotext_filename = Path(temp_imgtotext_filename)

            if temp_imgtotext_filename.is_file() and temp_imgtotext_filename.stat().st_size > 0:
                file_status = "Data is extracted from the technical datasheet of "+ tds + " and stored in \"util / data / extractedinfo / textualdata\" folder"
            print("CV file_status: ", file_status)
            return file_status
    

    def extract_and_save_text(self, manufacturer, tds):

            file_status = ''
            matrix_extract_text = MatrixTextExtractor(self.prop_filename)
            
            #temp_imgtotext_filename =  matrix_extract_text.temp_imgtotext_dir + matrix_extract_text.filename.strip(".pdf") + "_preprocessed.txt"
            ''' Pre-processing by Opencv. Use of PyTessaract to extract text and save in database'''
            temp_imgtotext_filename = matrix_extract_text.pdf_to_text(manufacturer, tds)
            print("CV temp_imgtotext_filename: ", temp_imgtotext_filename)
            temp_imgtotext_filename = Path(temp_imgtotext_filename)

            if temp_imgtotext_filename.is_file() and temp_imgtotext_filename.stat().st_size > 0:
                file_status = "Data is extracted from the technical datasheet of "+ tds + " and stored in \"util / data / extractedinfo / textualdata\" folder"
            print("CV file_status: ", file_status)
            return file_status
            
            

    def extract_text_by_key(self, productname, firsttext, secondtext):

        matrix_extract_text = MatrixTextExtractor(self.prop_filename)
        manufacturer = matrix_extract_text.manufacturer
        product = productname

        #matrix_extract_text.text_preprocessing()

        ''' Regular Expression '''
        searchkey, searchvalue = matrix_extract_text.extract_text_by_key(productname, firsttext, secondtext)
            
        payload = {
            'manufacturer': manufacturer,
            'product': product,
            'searchkey': searchkey,
            'searchvalue': searchvalue
        }
        print("CV Payload:", payload)
        return payload

    def read_file(self, productname):
        matrix_extract_text = MatrixTextExtractor(self.prop_filename)
        text_file =  matrix_extract_text.temp_imgtotext_dir + productname + "_preprocessed.txt"
            
        with open(text_file, 'r') as f:
            file_content = f.read()
        return file_content
        
    '''--------------------Tabular Data Operation--------------------'''

    def infer_table(self, manufacturer, tds):

        file_status = ''
        matrix_extract_text = MatrixTextExtractor(self.prop_filename)

        ''' Pre-processing by Opencv. Now use of PyTessaract to extract text from those images and save in database'''
        inferred_img_status = matrix_extract_text.infer_table_on_tds(self.prop_filename, manufacturer, tds)           
        return inferred_img_status
    
    '''--------------------Extract Tabular Data in CSV format from Table images and Store all table images in a folder--------------------'''

    def extract_table_in_csv(self, manufacturer, tds):
        matrix_extract_text = MatrixTextExtractor(self.prop_filename)
        get_tab_img_set = matrix_extract_text.get_table_imglist(manufacturer, tds)
        return get_tab_img_set
        
    def store_all_table_img(self, manufacturer, tds):
        matrix_extract_text = MatrixTextExtractor(self.prop_filename)
        table_img_set = matrix_extract_text.merge_table_img(manufacturer, tds)
        return table_img_set


    '''--------------------Transfer CSV data to MongoDB--------------------'''

    def transfer_csv_data(self, manufacturer, tds):
        matrix_extract_text = MatrixTextExtractor(self.prop_filename)
        get_tab_img_set = matrix_extract_text.transfer_csv_to_db(manufacturer, tds)
        return get_tab_img_set
        
        
            