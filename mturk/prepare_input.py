import csv
import json
import numpy as np
import pandas as pd
import requests

INSTANCES_PER_PROBLEM = 10
SAMPLES_PER_LABEL = 30
NUM_INSTANCES_SMALL = 30
SEED = 1337

CONTENT_FILTER_REGEX = 'vent|nsfw'

np.random.seed(SEED)
df = pd.read_parquet('../preprocessed/vent-robust-splits.parquet')
cats = pd.read_csv('../preprocessed/category_names.csv')['name']
cats_sorted = cats.sort_values().reset_index(drop=True).reset_index()
cats_sorted.columns = ['emotion_index', 'emotion']

test_df = df[df.split == 'test']
test_no_vent_df = test_df[~df['text'].str.contains(CONTENT_FILTER_REGEX, case=False)]
test_shuffle_df = test_no_vent_df.sample(frac=1).reset_index(drop=True)
test_sample = test_shuffle_df.groupby('emotion_index').head(SAMPLES_PER_LABEL)
test_sample_cols = test_sample[['text', 'emotion_index']]

with_names = cats_sorted.merge(test_sample_cols, on='emotion_index', how='inner')[['emotion', 'text']]
with_names_shuffle = with_names.sample(frac=1).reset_index(drop=True)
as_grouping = with_names_shuffle.groupby(with_names_shuffle.index // INSTANCES_PER_PROBLEM)
as_json_groups = as_grouping.apply(lambda group: [{'emotion': e, 'text': t} for e, t in zip(group.emotion, group.text)])

with open('input.csv', 'w') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(['INSTANCE_DATA'])
    for json_obj in as_json_groups:
        writer.writerow([json.dumps(json_obj)])

with open('input_small.csv', 'w') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(['INSTANCE_DATA'])
    for i, json_obj in enumerate(as_json_groups):
        writer.writerow([json.dumps(json_obj)])
        if i == NUM_INSTANCES_SMALL - 1:
            break
