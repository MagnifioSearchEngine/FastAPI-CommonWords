from fastapi import FastAPI
from pydantic import BaseModel
from pymongo import MongoClient, mongo_client
import pandas as pd

from lambda_function import create_tfidf_features, calculate_similarity, show_similar_documents, preprocess

mongo_client = MongoClient(
    "mongodb+srv://Pranay:confidential@cluster0.2bumc.mongodb.net/commonwords?retryWrites=true&w=majority")

mongo_database = mongo_client['upload_files']
temp_collection = mongo_database['temp']
files_collection = mongo_database['files']
common_collection = mongo_database['common']
cache_collection = mongo_database['cache']

app = FastAPI()

class Key(BaseModel):
    key: str

@app.get('/')
def get():
    return {
        "Success": True
    }

@app.post('/')
def read(body: Key):
    db_temp = temp_collection.find_one({ "key": body.key }, { "_id": 0 })
    db_file = files_collection.find({ "company": body.key }, { "_id": 0 })
    db_common = common_collection.find_one({ "key": body.key }, { "_id": 0 })

    data_frame = pd.DataFrame(list(db_file))
    print(type(db_temp['data'][0]))

    data = [preprocess(title, content)
     for title, content in zip([""]*len(data_frame["text"]), data_frame['text'])]

    X, v = create_tfidf_features(db_temp['data'], data_frame)

    for topic in db_common['common']:
        user_question = [topic]
        sim_vecs, cosine_similarities = calculate_similarity(
            X, v, user_question)
        output = show_similar_documents(data, cosine_similarities, sim_vecs)
        newDict = {
            "company": body.key,
            "keyword": topic,
            "search_results": output
        }
        # print(newDict)

        cache_collection.insert_one(newDict)

    return {
        "Success": True,
        "data": "db_data"
    }
