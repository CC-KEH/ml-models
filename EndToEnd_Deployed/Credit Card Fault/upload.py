from pymongo.mongo_client import MongoClient
import pandas as pd
import json
from src.constant import MONGO_URI,MONGO_DATABASE_NAME,MONGO_COLLECTION_NAME

# uniform resource indentifier
uri = MONGO_URI
print(uri)
# Create a new client and connect to the server
client = MongoClient(uri)

# create database name and collection name
DATABASE_NAME=MONGO_DATABASE_NAME
COLLECTION_NAME=MONGO_COLLECTION_NAME

dataset = pd.read_csv('notebooks/data/ccf.csv')
json_record = list(json.loads(dataset.T.to_json()).values())

#now dump the data into the database
client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)