from pymongo.mongo_client import MongoClient
import pandas as pd 
import json

# Uniform Resource Identifier
uri="mongodb+srv://Luciferr:esperantO@2.0@cluster1.8a1b2p2.mongodb.net/?retryWrites=true&w=majority"

#create a new client and connect to the server
client = MongoClient(uri)

#create database name and collection
DATABASE_NAME = 'ML_project'
COLLECTION_NAME = 'WaferFaultDetection'

#read the data as a dataframe
df = pd.read_csv(r)
df=df.drop('Unnamed:0',axis=1)

#convert data into json
json_record=list(json.loads(df.T.to_json()).values())

#Dumping data into Database
client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)


