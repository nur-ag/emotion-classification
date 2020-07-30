import re
import numpy as np
import pandas as pd

df = pd.read_csv('../data/GoEmotions.tsv', delimiter='\t')
names_handle = open('../data/GoEmotionsNames.txt', 'r')
emotion_mapping = {i: em.strip() for i, em in enumerate(names_handle)}

num_emotions = len(emotion_mapping)
eye = np.eye(num_emotions)
df.emotion_ids = [[int(em_id) for em_id in emotion_string.split(',')]
				    		  for emotion_string in df.emotion_ids]
df['emotions_label'] = [eye[ids].sum(axis=0) for ids in df.emotion_ids]
df['emotions'] = [[emotion_mapping[em_id] for em_id in emotion_ids]
										  for emotion_ids in df.emotion_ids]

df.to_parquet('../preprocessed/GoEmotions.parquet')
