# Copyright (c) ArnabGhoshChowdhury, Universität Osnabrück and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from django.db import models
from django.db.models.fields import BigAutoField

# Create your models here.
class Datasheet(models.Model):
    manufacturer_id = models.TextField()
    manufacturer =  models.TextField()
    tdslist = models.TextField()
    
    class MongoMeta:
        db_table = "datasheet"

    def __str__(self):
        return self.manufacturer + " : " + self.tdslist

class TableInfo(models.Model):
    manufacturer = models.TextField()
    tdsname = models.TextField()
    tabledata = models.TextField(default=None, blank=True, null=True)
   
    class MongoMeta:
        db_table = "tableinfo"


    def __str__(self):
        return str(self.manufacturer + " -- " + self.tdsname)