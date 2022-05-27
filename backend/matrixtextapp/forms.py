# Copyright (c) ArnabGhoshChowdhury, Universität Osnabrück and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from django import forms
import os
from django.conf import settings
import xml.etree.ElementTree as ET
from .models import Datasheet, TableInfo

prop_filename = os.path.join( settings.BASE_DIR,'util/prop/MDE.xml')
root = ET.parse(prop_filename).getroot()

class DocListForm(forms.Form):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    class Meta: 
        model = Datasheet
        fields = '__all__'

class MergeImgFileForm(forms.Form):
    manufacturer = forms.CharField(label="manufacturer")

class DataExtractForm(forms.Form):
    manufacturer = forms.CharField(label="manufacturer")
    tdslist = forms.CharField(label="tdslist")


class SearchTextForm(forms.Form):
    firsttext = forms.CharField(label="firsttext")
    secondtext = forms.CharField(label="secondtext")
    productname = forms.CharField(label="productname")

class EsDataForm(forms.Form):
    manufacturer = forms.CharField(label="manufacturer", required = False)
    tdsname = forms.CharField(label="tdsname", required = False)
    tabledata = forms.CharField(label="tabledata", required = False)