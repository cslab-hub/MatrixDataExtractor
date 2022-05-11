# Copyright (c) ArnabGhoshChowdhury, Universität Osnabrück and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
from pymongo import MongoClient
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from collections import deque
from tqdm import tqdm
import time


from elasticsearch_dsl import Search, Q

def mongoconnect(db_name, coll_name):
    mongo_client = MongoClient("mongodb://localhost:27017/")
    mong_db = mongo_client[db_name]
    mongo_coll = mong_db[coll_name]
    return mong_db, mongo_coll


# Pull from mongo and dump into ES using bulk API
def load_data_to_es(elastic_client, mongo_db, mongo_coll):
    esinfo=(elastic_client.info())
    #print("esinfo: ", esinfo)

    actions = []
    for data in tqdm(mongo_coll.find(), total=mongo_coll.estimated_document_count()):
        data.pop('_id')
        action = {
                    "index": {
                        "_index": 'tabinfo',
                        "_type": 'tds',
                    }
                }
        actions.append(action)
        actions.append(data)

    delete = elastic_client.indices.delete(index = 'tabinfo')
    request_body = {
        "settings" : {
            "number_of_shards": 1,
            "number_of_replicas": 0
        }
    }
    elastic_client.indices.create(index='tabinfo', body = request_body, ignore=400)
    res = elastic_client.bulk(index = 'tabinfo', body = actions, refresh = True)
    bulk_result = elastic_client.search(index='tabinfo', body={"query": {"match_all": {}}}, size=50)['hits']['hits']
    
    #for doc in elastic_client.search(index='tabinfo', body={"query": {"match_all": {}}}, size=50)['hits']['hits']:
    #    print("SOURCE\n", doc['_source'])
    
    return elastic_client   

def search_query(es_client, manf_name="", tds_name="", table_data=""):  
        search=list()
        if manf_name!="" and tds_name!="" and table_data!="":  
            q = Q("bool", must=[Q("match", manufacturer=manf_name), 
                              Q("match", tds=tds_name),
                              Q("match", tabular_data=table_data),
                            ]) 

        elif manf_name!="" and tds_name!="":
            q = Q("bool", must=[Q("match", manufacturer=manf_name), 
                              Q("match", tds=tds_name),
                            ])

            s = Search(using=es_client, index="tabinfo").query(q)[0:20]
            response = s.execute()
            search = get_results(response)

        elif manf_name!="" and table_data!="":
            q = Q("bool", must=[Q("match", manufacturer=manf_name), 
                              Q("match", tabular_data=table_data),
                            ])

            s = Search(using=es_client, index="tabinfo").query(q)[0:20]
            response = s.execute()
            print("response: ", response)
            search = get_results(response)

        elif tds_name!="" and table_data!="":
            q = Q("bool", must=[Q("match", tds=tds_name), 
                              Q("match", tabular_data=table_data),
                            ])

            s = Search(using=es_client, index="tabinfo").query(q)[0:20]
            response = s.execute()
            print("response: ", response)
            search = get_results(response)

        else:
            #print("manf_name, tds_name, table_data: ", manf_name, tds_name, table_data)
            q = Q("bool", should=[Q("match", manufacturer=manf_name), 
                              Q("match", tds=tds_name),
                              Q("match", tabular_data=table_data),
                            ], minimum_should_match=1) 
            s = Search(using=es_client, index="tabinfo").query(q)[0:20]
            response = s.execute()
            #print("response: ", response)
            search = get_results(response)

        return search

def get_results(response):
        results = []
        for hit in response:
            result_tuple = (hit.manufacturer, hit.tds, hit.tabular_data)
            results.append(result_tuple)
        return results

def uos_esearch(manf_name="", tds_name="", table_data=""):
    
    mongo_db, mongo_coll = mongoconnect("matrixtextapp", "matrixtextapp_tabledata")
    elastic_client = Elasticsearch(hosts=["localhost"],port=9200)
    es_client = load_data_to_es(elastic_client, mongo_db, mongo_coll)  
    es_result = search_query(es_client, manf_name, tds_name, table_data)
    return es_result
    
    
