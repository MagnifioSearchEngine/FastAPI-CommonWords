from fastapi import FastAPI
from pydantic import BaseModel
from pymongo import MongoClient
import pandas as pd
import uvicorn

from lambda_function import create_tfidf_features, calculate_similarity, show_similar_documents, preprocess

client = MongoClient(
    "mongodb+srv://Pranay:confidential@cluster0.2bumc.mongodb.net/commonwords?retryWrites=true&w=majority")
db = client["upload_files"]
files = db.files
cache = db.cache

app = FastAPI()

class Key(BaseModel):
    company: str
    topics: list

@app.get('/')
def get():
    return {
        "Success": True
    }



@app.post('/common')
def read(body: Key):
    
    df = pd.DataFrame(list(files.find({"company": body.company})))
    

    if df:
        data = [preprocess(title, content) for title, content in zip([""]*len(df["text"]), df['text'])]
        X,v = create_tfidf_features(data,df)
        for topic in body.topics:
                user_question = [topic]
                sim_vecs, cosine_similarities = calculate_similarity(X,v, user_question)
                output = show_similar_documents(data, cosine_similarities, sim_vecs)
                newDict={
                    "company":company,
                    "keyword":topic,
                    "search_results":output
                    }
                    
                cache.insert_one(newDict)
            return{"message:success"}

    return {
        "Success": True,
        "data": "db_data"
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)
