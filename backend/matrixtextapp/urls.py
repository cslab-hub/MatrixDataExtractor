from os import name
from django.contrib import admin
from django.urls import path, include 
from matrixtextapp import views


urlpatterns = [
   path('',views.index, name="index"),
   path('syncdata/',views.syncdata, name="syncdata"),
   path('textextraction/',views.text_extraction, name="textextraction"),  
   path('tableextraction/',views.table_extraction, name="tableextraction"),  
   path('extracttabledata/', views.extract_table_data, name="extracttabledata"),
   path('transfertabledata/', views.transfer_table_data, name="transfertabledata"),
   path('tableinfosearch/', views.table_info_search, name="tableinfosearch")
   
]
