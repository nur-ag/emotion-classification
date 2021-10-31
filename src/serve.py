import tqdm

# Monkey patch tqdm so that we don't get progress bars in inference
def nop(it, *a, **k):
    return it
tqdm.tqdm = nop

# Import everything after the monkey patching
import dill
import json
import glob
import torch

from utils.release import ReleaseModel

import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Redefine the maximum length of an input string
MAX_VALID_LENGTH = 1024
NUM_MAX_TOKENS = 256

# Monkey patch models that were trained in GPU to load in non-GPU envs
if not torch.cuda.is_available():
    base_load = torch.load
    torch.load = lambda f: base_load(f, map_location='cpu')

class PredictionRequest(BaseModel):
    text: str
    model: str

app = FastAPI()
origins = [
    "http://emotionui-anon.s3-website-us-west-1.amazonaws.com",
    "http://emotionui.nur.systems",
    "http://localhost",
    "http://localhost:8080",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

models = {}
for path in glob.glob('release/*.pkl'):
    with open(path, 'rb') as fp:
        rm = dill.load(fp)
        models[rm.name] = rm


@app.get("/")
def root():
    return {'info': 'Send a POST request to the `predict` endpoint with `text` and `model` arguments.'}


@app.post("/predict/")
async def predict(prediction: PredictionRequest):
    rm = models.get(prediction.model, None)
    if rm is None:
        return {'error': f'Unknown model "{prediction.model}".'}

    representation = rm.extractor([prediction.text[:MAX_VALID_LENGTH]])
    emotions = rm.model.predict([representation]).flatten().tolist()
    result_objects = []
    for label, score, threshold in sorted(zip(rm.labels, emotions, rm.thresholds), key=lambda x: -x[1]):
        category = rm.category_dict.get(label, 'Unknown')
        result_objects.append({
                'id': rm.get_id(label),
                'emotion': label,
                'score': score,
                'threshold': threshold,
                'active': bool(score > threshold),
                'category': category,
                'color': rm.color_dict.get(category, '#000000')
            })
    return {'text': prediction.text, 
            'model': prediction.model, 
            'emotions': result_objects}
