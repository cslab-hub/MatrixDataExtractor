# Copyright (c) ArnabGhoshChowdhury, Universität Osnabrück and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from curses.ascii import NUL
from typing import Mapping, Text
from django.http import response
from django.http import request
from django.http.response import HttpResponse, HttpResponseRedirect, JsonResponse
from django.http.request import HttpRequest
from django.shortcuts import render
from rest_framework.response import Response
from rest_framework import serializers, status, mixins, generics
from rest_framework.decorators import api_view
from django.conf import settings
from django.core.serializers import serialize
from django.core.serializers.json import DjangoJSONEncoder
from django.http import Http404
from .models import Datasheet, TableInfo
from .serializers import DatasheetSerializer, TabularInfoSerializer
from .forms import DataExtractForm, DocListForm,EsDataForm
from .cv_basic_service import MatrixInfoExtractorManager, MatrixTextExtractor
import os
import json, datetime, pymongo
from pymongo import MongoClient
import xml.etree.ElementTree as ET
from ast import literal_eval
from django.db.models.functions import Coalesce
from .es_call import uos_esearch


# Create your views here.
prop_filename = os.path.join( settings.BASE_DIR,'util/prop/MDE.xml')
xml_root = ET.parse(prop_filename).getroot()

def mongoconnect(db_name, coll_name):
   client = MongoClient("mongodb://localhost:27017/")
   db = client[db_name]
   coll = db[coll_name]
   return db, coll
'''
mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
mongo_db = mongo_client["matrixtextapp"]
mongo_coll = mongo_db["matrixtextapp_datasheet"]
'''

mongo_db, mongo_coll = mongoconnect("matrixtextapp", "matrixtextapp_datasheet")
# Create MongoDB Compound Index on manufacturer_id, manufacturer, tdslist
# src: https://www.analyticsvidhya.com/blog/2020/09/mongodb-indexes-pymongo-tutorial/
mongo_db.matrixtextapp_datasheet.create_index([('manufacturer_id',pymongo.ASCENDING),
                                                ('manufacturer',pymongo.ASCENDING),
                                                ('tdslist', pymongo.ASCENDING)],
                                                name='manufacturer_tdslist')


'''--------------------Index Page--------------------'''

def index(request):
   return render(request, "site/index.html")

def syncdata(request):
   manufacturer_list = []
   file_list = []
   doclist_instances = ''
   if request.method == 'POST':
      if request.POST.get("setparam"):
         pass

      elif request.POST.get('showtds',''):
         form = DocListForm(request.POST or None)
         if form.is_valid():
            manufacturer_doc_count = mongo_coll.delete_many({})
            manufacturer_id = 1
            
            for child_element in xml_root.findall('dataset/tabledetection'):
               srcpdf_filepath = child_element.find('sourcepdf').text
               srcpdf_filepath = os. getcwd() + srcpdf_filepath
               for (root,dirs,files) in os.walk(srcpdf_filepath, topdown=True):
                  if len(dirs)!=0:
                     manufacturer_list = dirs
                     manufacturer_list.sort()
                  if len(files) != 0:
                        for manufacturer_name in manufacturer_list:
                           if manufacturer_name == str(root.rsplit('/',1)[1]):
                              files.sort()
                              file_list.append(files)
                              doclist_instances = Datasheet.objects.create(manufacturer_id = manufacturer_id,manufacturer=manufacturer_name, tdslist=files)
                              manufacturer_id = manufacturer_id+1
   return render(request, "site/syncdata.html", {})

'''--------------------All Data Extraction in Text format--------------------'''
'''
Table Region and then extract tabular data
'''
def get_table_img_info():
   manufacturer_list = []
   tabimg_tds_dict = {}
   tds_info = []
   tds_new_dict = {}
   new_tds_key = "None"
   new_tds_value = list()
   tds_dict = {}
   
            
   for child_element in xml_root.findall('dataset/tabledetection'):
    srcpdf_filepath = child_element.find('infertableimg').text
    srcpdf_filepath = os. getcwd() + srcpdf_filepath
    
    for manufacturer in os.listdir(srcpdf_filepath):
        tds_list = []
        manufacturer_list.append(manufacturer)
        manufacturer_list.sort()
        for dirs in os.listdir(os.path.join(srcpdf_filepath,manufacturer)):
            if len(dirs) != 0: 
               tds_list.append(dirs)
               tds_list.sort()
        tabimg_tds_dict = {'manufacturer': manufacturer, 'tdslist': tds_list}
        tds_info.append(tabimg_tds_dict)
        
    for i in range(len(tds_info)):
      for k, v in tds_info[i].items():
        if(k == "manufacturer"):
            new_tds_key = tds_info[i]["manufacturer"]
      
        elif (k == "tdslist"):
            new_tds_value = tds_info[i]["tdslist"]
           
        tds_dict[new_tds_key] = new_tds_value
   tds_dict.pop('None', None)
   tds_new_dict = json.dumps(tds_dict)   
   return tds_new_dict   


'''
Return Manufacturer List and Technical Datasheet Lists from MongoDB database
'''
def get_tds_new_dict():
   tds_dict = {}
   tds_info = Datasheet.objects.values()
   new_tds_value = list()
   new_tds_key = "None"
   for i in range(len(tds_info)):
      for k, v in tds_info[i].items():
         #print(k ,"::", v)
         if(k == "manufacturer"):
            new_tds_key = tds_info[i]["manufacturer"]
      
         elif (k == "tdslist"):
            new_tds_value = tds_info[i]["tdslist"]
            #print("new_tds_value: ", type(new_tds_value))
         tds_dict[new_tds_key] = new_tds_value
   tds_dict.pop('None', None)
   tds_new_dict = json.dumps(tds_dict)
   return tds_new_dict

def text_extraction(request):
   if request.method == 'GET':
      tds_new_dict = get_tds_new_dict()
      return render(request, 'site/textextraction.html', {"tds_new_dict": tds_new_dict})

   file_status = ''
   if request.method == 'POST':
      tds_new_dict = get_tds_new_dict()
      if request.POST.get("setparam"):
         pass

      elif request.POST.get('extractdata',''):
         form = DataExtractForm(request.POST or None)
         if form.is_valid():
            manufacturer = form.cleaned_data["manufacturer"]
            tds = form.cleaned_data["tdslist"]
            prop_filename = os.path.join( settings.BASE_DIR,'util/prop/MDE.xml')
            matrix_ie_manager = MatrixInfoExtractorManager(prop_filename)      
            file_status = matrix_ie_manager.extract_and_save_text(manufacturer, tds)
            print("file_status: ", file_status)
      return render(request, "site/textextraction.html", {"tds_new_dict": tds_new_dict, "file_status": file_status})
 
         
'''--------------------Transfer Tabular Data to Destination--------------------'''

def transfer_table_data(request):
   tds_new_dict = get_table_img_info()
   if request.method == 'GET':
      return render(request, 'site/transfertabledata.html', {"tds_new_dict": tds_new_dict})

   file_status = ''
   if request.method == 'POST':
      if request.POST.get("setparam"):
         pass

      elif request.POST.get('gettableimage',''):
         form = DataExtractForm(request.POST or None)
         if form.is_valid():
            manufacturer = form.cleaned_data["manufacturer"]
            tds = form.cleaned_data["tdslist"]
            prop_filename = os.path.join( settings.BASE_DIR,'util/prop/MDE.xml')
            matrix_ie_manager = MatrixInfoExtractorManager(prop_filename)      
            table_img_list = matrix_ie_manager.transfer_csv_data(manufacturer, tds) 

      return render(request, "site/transfertabledata.html",{"tds_new_dict": tds_new_dict, "table_img_list": table_img_list})
            
'''--------------------Tabular Data Extraction--------------------'''

def extract_table_data(request):
   tds_new_dict = get_table_img_info()  
   print("tds_new_dict: ", tds_new_dict)
   if request.method == 'GET':  
      return render(request, 'site/extracttabledata.html', {"tds_new_dict": tds_new_dict})

   file_status = ''
   if request.method == 'POST':
      if request.POST.get("setparam"):
         pass

      elif request.POST.get('gettableimage',''):
         form = DataExtractForm(request.POST or None)
         if form.is_valid():
            manufacturer = form.cleaned_data["manufacturer"]
            tds = form.cleaned_data["tdslist"]
            prop_filename = os.path.join( settings.BASE_DIR,'util/prop/MDE.xml')
            matrix_ie_manager = MatrixInfoExtractorManager(prop_filename)      
            table_img_set = matrix_ie_manager.extract_table_in_csv(manufacturer, tds) 

      return render(request, "site/extracttabledata.html",{"tds_new_dict": tds_new_dict, "table_img_set": table_img_set})



'''--------------------Table Region Detection--------------------'''

def table_extraction(request):
   tds_new_dict = get_tds_new_dict()
   if request.method == 'GET':
      return render(request, 'site/tableextraction.html', {"tds_new_dict": tds_new_dict})
      

   file_status = ''
   if request.method == 'POST':
      if request.POST.get("setparam"):
         pass

      elif request.POST.get('extracttable',''):
         form = DataExtractForm(request.POST or None)
         if form.is_valid():
            manufacturer = form.cleaned_data["manufacturer"]
            tds = form.cleaned_data["tdslist"]

            print("Arnab_manufacturer_tds: ", manufacturer, " ::: ", tds)

            prop_filename = os.path.join( settings.BASE_DIR,'util/prop/MDE.xml')
            matrix_ie_manager = MatrixInfoExtractorManager(prop_filename)      
            table_extract_status = matrix_ie_manager.infer_table(manufacturer, tds)
            print("table_extract_status: ", table_extract_status)
      return render(request, "site/tableextraction.html", {"tds_new_dict": tds_new_dict, "table_extract_status": table_extract_status})
 
         

'''--------------------ES Search Index-------------------'''

def table_info_search(request):
   
   manufacturer = ""
   tdsname = ""
   tabledata = ""

   if request.method == 'GET':
      return render(request, 'site/tableinfosearch.html', {})
      

   if request.method == 'POST':
      if request.POST.get("setparam"):
         pass

      elif request.POST.get('gettabinfo',''):
         form = EsDataForm(request.POST or None)
         if form.is_valid():
            manufacturer = form.cleaned_data["manufacturer"]
            tdsname = form.cleaned_data["tdsname"]
            tabledata = form.cleaned_data["tabledata"]
            
            
            #print("Arnab_tabinfo: ", manufacturer, " :: ", tdsname, "::", tabledata)
            es_result = uos_esearch(manufacturer, tdsname, tabledata)
            len_result = len(es_result)
            print("es_result: ", es_result)
            

      return render(request, "site/tableinfosearch.html", {"count": len_result,"es_result": es_result})
'''--------------------End of Views--------------------'''



