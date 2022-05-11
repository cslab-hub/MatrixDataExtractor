# Copyright (c) ArnabGhoshChowdhury, Universität Osnabrück and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from rest_framework import serializers
from .models import Datasheet, TableInfo

class DatasheetSerializer(serializers.ModelSerializer):
    class Meta:
        model = Datasheet
        fields = '__all__' 

class TabularInfoSerializer(serializers.ModelSerializer):
    class Meta:
        model = TableInfo
        fields = '__all__'
