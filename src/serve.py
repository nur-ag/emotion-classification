import dill
from pydantic import BaseModel
from fastapi import FastAPI

class PredictionRequest(BaseModel):
    text: str
    model: str

app = FastAPI()

with open('../models/GoEmotions/test.pkl', 'rb') as fp:
    labels, extractor, model = dill.load(fp)

def get_model(model_name):
    return extractor, model 


@app.get("/")
def root():
    return {'info': 'Send a POST request to the `predict` endpoint with `text` and `model` arguments.'}


@app.post("/predict/")
async def predict(prediction: PredictionRequest):
    extractor, model = get_model(prediction.model)
    representation = extractor([prediction.text]).reshape(1, -1)
    emotions = model.predict([representation]).flatten().tolist()
    emos_labelled = {label: score for label, score in zip(labels, emotions)}
    return {'text': prediction.text, 
            'model': prediction.model, 
            'emotions': emos_labelled}


