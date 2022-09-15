import numpy as np
import random, os
from pdf2image import convert_from_path
import xml.etree.ElementTree as ET
import pdfplumber
from PIL import Image
import pandas as pd
from pathlib import Path
import camelot

np.random.seed(42)
random.seed(42)


class MatrixTableDataExtractor(object):

    def __init__(self, prop_filename):

        '''----------Extract data from XML Configuration File----------'''
        root = ET.parse(prop_filename).getroot()
        for child_element in root.findall('dataset/tabledetection'):
            '''
            Current Directory + Path of other dirs retrieved from XML files
            '''
            self.pdf_dirname = os.getcwd()  + child_element.find('sourcepdf').text
            self.infer_imgdir = os.getcwd() + child_element.find('inferimgdir').text
            self.bbox_dir = os.getcwd()  + child_element.find('bboxinfo').text
            self.temp_img_dir = os.getcwd()  + child_element.find('tempimgdir').text
            self.table_data_dir = os.getcwd() + child_element.find('tabulardata').text
            
            
    def get_pdf_dim(self, pdf_path):
        with pdfplumber.open(pdf_path) as pdf:
            first_page = pdf.pages[0]
            return int(first_page.width), int(first_page.height)
    
    '''Get document image dimension'''
    def get_img_dim(self, temp_doc_img_dir, imgfile_wo_extension):

        '''
        for row in doc_img_df.itertuples():
            doc_img_list.append(row.DOC_IMG)
        for doc_img in doc_img_list:   
            first_img_file = doc_img + '.jpg'
            temp_img = os.path.join(temp_img_dir,first_img_file)
            img= Image.open(temp_img)
            img_width, img_height = img.size
        '''
        imgfile_w_extension = imgfile_wo_extension + '.jpg'
        first_doc_img = os.path.join(temp_doc_img_dir,imgfile_w_extension)
        img = Image.open(first_doc_img)
        img_w, img_h = img.size
        return img_w, img_h


    def get_img_bbox_info(self, filename_wo_extension):
        csv_file = filename_wo_extension + '.csv'
        bbox_info_file = os.path.join(self.bbox_dir,csv_file)
        colnames=['DOC_IMG', 'IMG_NO', 'X_MIN', 'Y_MIN','X_MAX','Y_MAX'] 
        doc_img_df = pd.read_csv(bbox_info_file, sep='\t', names=colnames, header=None, index_col=None)
        return doc_img_df


    def extract_tabulardata_from_pdf(self, filename):
        Path(self.table_data_dir).mkdir(parents=True, exist_ok=True)            

        if filename.lower().endswith('.pdf'):
            filename_wo_extension = os.path.splitext(filename)[0]
            pdf_path = os.path.join(self.pdf_dirname,filename)
            filenamedir = os.path.join(self.table_data_dir,filename_wo_extension)
            temp_doc_img_dir = os.path.join(self.temp_img_dir,filename_wo_extension)
                    
            if os.path.isfile(pdf_path):
                Path(filenamedir).mkdir(parents=True, exist_ok=True)
                '''-----Get scale value: scale_w = (img_w/pdf_w) and scale_h = (img_h/pdf_h)-----'''
                '''Get dimension of first page of PDF document and first document_image'''
                pdf_w, pdf_h = self.get_pdf_dim(pdf_path)
                
                '''Get bbox information'''
                doc_img_df = self.get_img_bbox_info(filename_wo_extension)
                '''Map to PDF for each document image'''
                # https://note.nkmk.me/en/python-pandas-dataframe-for-iteration/
                for bbox_info_row in doc_img_df.itertuples():
                    #print(bbox_info_row.DOC_IMG, ",", bbox_info_row.IMG_NO, ",",bbox_info_row.X_MIN,",",bbox_info_row.Y_MIN,",",bbox_info_row.X_MAX,",",bbox_info_row.Y_MAX)
                    '''Doc_img name without extension'''
                    imgfile_wo_extension = bbox_info_row.DOC_IMG
                    table_no_on_single_pg = str(bbox_info_row.IMG_NO)
                    '''Reminder: PDF page starts with Index=0 in MDE. But for Camelot, it starts with 1. So for camelot, pages=str(1+pdf_pg_no)'''
                    pdf_pg_no = int(imgfile_wo_extension.split('_')[-1])
                    #print("imgfile_wo_extension: ", imgfile_wo_extension, ",", pdf_pg_no)
                    '''Doc_img X co-ordinates'''
                    img_x_min = bbox_info_row.X_MIN
                    img_x_max = bbox_info_row.X_MAX

                    '''Doc_img Y co-ordinates'''
                    img_y_min = bbox_info_row.Y_MIN
                    img_y_max = bbox_info_row.Y_MAX

                    '''Doc_img dimension and calculate scale w.r.t. PDF dimension'''
                    img_w, img_h = self.get_img_dim(temp_doc_img_dir,imgfile_wo_extension)
                    scale_w = int(img_w/pdf_w)
                    scale_h = int(img_h/pdf_h)
                    
                    '''Calculate PDF X and Y co-ordinates: pdf_x = int(img_x/scale_w) and pdf_y = int((img_h-img_y)/scale_h)'''
                    pdf_table_x_min = int(img_x_min/scale_w)
                    pdf_table_x_max = int(img_x_max/scale_w)

                    pdf_table_y_min = int((img_h-img_y_min)/scale_h)
                    pdf_table_y_max = int((img_h-img_y_max)/scale_h)

                    pdf_table_coordinate = str(int(pdf_table_x_min)) +","+ str(int(pdf_table_y_min))+ ","+ str(int(pdf_table_x_max))+ ","+ str(int(pdf_table_y_max))
                    pdf_table = camelot.read_pdf(pdf_path, pages=str(1+pdf_pg_no), flavor='stream',  table_areas=[pdf_table_coordinate])
                    df_table = pdf_table[0].df
                    '''Save tabular data'''
                    table_excelfile = filenamedir + "/" + imgfile_wo_extension + '_' + table_no_on_single_pg + ".xlsx"
                    df_table.to_excel(table_excelfile)

            else:
                print("PDF document not found.")


if __name__ == "__main__":
    print("Program starts !", flush=True)
    prop_filename = 'util/prop/coac_prop.xml'
    mtdep = MatrixTableDataExtractor(prop_filename)
    ''' Pre-process all PDF documents in srcpdf folder '''
    #mtdep.all_pdf_to_img()
    ''' Pre-process only single PDF document in srcpdf folder '''
    #tdsname = 'Your folder name from coac/util/data/srcpdf/Filename.pdf'
    tdsname = 'Petrothene NA940000.pdf'
    mtdep.extract_tabulardata_from_pdf(tdsname)
    print("Program ends !", flush=True)