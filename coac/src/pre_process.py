import os
from pathlib import Path
from numpy.core.fromnumeric import product
from pdf2image import convert_from_path
import xml.etree.ElementTree as ET
import numpy as np
import random

np.random.seed(42)
random.seed(42)


class MatrixTableDataExtractPreprocessor(object):

    def __init__(self, prop_filename):

        '''----------Extract data from XML Configuration File----------'''
        root = ET.parse(prop_filename).getroot()
        for child_element in root.findall('dataset/tabledetection'):
            '''
            Current Directory + Path of other dirs retrieved from XML files
            '''
            self.pdf_dirname = os.getcwd()  + child_element.find('sourcepdf').text
            self.temp_imgdir = os.getcwd()  + child_element.find('tempimgdir').text
            self.infer_imgdir = os.getcwd() + child_element.find('inferimgdir').text
            
    def all_pdf_to_img(self) :
            
            Path(self.pdf_dirname).mkdir(parents=True, exist_ok=True)
            Path(self.temp_imgdir).mkdir(parents=True, exist_ok=True)
            Path(self.infer_imgdir).mkdir(parents=True, exist_ok=True)

            for filename in os.listdir(self.pdf_dirname):
                if filename.lower().endswith('.pdf'):
                    filename_wo_extension = os.path.splitext(filename)[0]
                    filenamedir = os.path.join(self.temp_imgdir,filename_wo_extension)
                    pdf_path = os.path.join(self.pdf_dirname,filename)
                    
                    if os.path.isfile(pdf_path):
                        Path(filenamedir).mkdir(parents=True, exist_ok=True)
                        ''' Split PDF and convert into images '''
                        pages = convert_from_path(pdf_path, dpi=72)
                        pg_no = 0
                        for page in pages:
                            page.save( self.temp_imgdir + "/" + filename_wo_extension + "/"+ filename_wo_extension+'_'+str(pg_no)+ '.jpg', 'JPEG')
                            pg_no = pg_no + 1
                            
    def single_pdf_to_img(self, filename) :
            
            Path(self.pdf_dirname).mkdir(parents=True, exist_ok=True)
            Path(self.temp_imgdir).mkdir(parents=True, exist_ok=True)
            Path(self.infer_imgdir).mkdir(parents=True, exist_ok=True)

            if filename.lower().endswith('.pdf'):
                filename_wo_extension = os.path.splitext(filename)[0]
                filenamedir = os.path.join(self.temp_imgdir,filename_wo_extension)
                pdf_path = os.path.join(self.pdf_dirname,filename)
                    
                if os.path.isfile(pdf_path):
                    Path(filenamedir).mkdir(parents=True, exist_ok=True)
                    ''' Split PDF and convert into images '''
                    pages = convert_from_path(pdf_path, dpi=72)
                    pg_no = 0
                    for page in pages:
                        page.save( self.temp_imgdir + "/" + filename_wo_extension + "/"+ filename_wo_extension+'_'+str(pg_no)+ '.jpg', 'JPEG')
                        pg_no = pg_no + 1

                else:
                    print("PDF document not found.")


if __name__ == "__main__":
    print("Program starts !", flush=True)
    prop_filename = 'util/prop/coac_prop.xml'
    mtdep = MatrixTableDataExtractPreprocessor(prop_filename)
    ''' Pre-process all PDF documents in srcpdf folder '''
    #mtdep.all_pdf_to_img()
    ''' Pre-process only single PDF document in srcpdf folder '''
    tdsname = 'Only Your Filename.pdf in String format, which is taken from coac/util/data/srcpdf/Filename.pdf' #'Petrothene NA940000.pdf'
    mtdep.single_pdf_to_img(tdsname)
    print("Program ends !", flush=True)
