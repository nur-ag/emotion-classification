import re
import numpy as np
import pandas as pd

from utils.tokens import filter_by_num_tokens, normalize_text

# Prepare the emotion categories by removing disabled and non-ascii categories
emotions_raw = pd.read_csv('../data/emotions.csv')
emotions_clean = emotions_raw[emotions_raw.enabled & emotions_raw.name.str.fullmatch('[a-zA-Z]+')]
emotions = emotions_clean.sort_values('name')
emotions.to_csv('../data/emotions_clean.csv')

clean_categories = emotions.emotion_category_id.drop_duplicates().to_frame().rename(columns={'emotion_category_id': 'id'})
emotion_categories_raw = pd.read_csv('../data/emotion_categories.csv')
emotion_categories_clean = emotion_categories_raw[emotion_categories_raw.name.str.fullmatch('[a-zA-Z]+')]
emotion_categories = emotion_categories_clean.merge(clean_categories, on='id', how='inner').sort_values('name')
emotion_categories.to_csv('../data/emotion_categories_clean.csv')

# Load the graph and text dataset
graph = pd.read_csv('../data/vent.edgelist', names=['source_user_id', 'dest_user_id'], sep=' ')
vents = pd.read_csv('../data/vents.csv')
vents.text = [normalize_text(text) for text in vents.text.astype(str)]
num_tokens_filter = lambda x: filter_by_num_tokens(x, 3, 32)
vents = vents[vents.text.apply(num_tokens_filter)].drop_duplicates(subset=['text'])

# Load the emotion and category names and their corresponding indices
emotion_names = {name: index for index, name in enumerate(emotions.name.tolist())}
category_names = {name: index for index, name in enumerate(emotion_categories.name.tolist())}

# Compute emotion category indices
emotion_categories.columns = ['emotion_category_id', 'emotion_category_name']
all_emotion_categories = sorted(emotion_categories.emotion_category_id.unique())
emotion_category_indices = {c: i for i, c in enumerate(all_emotion_categories)}
emotion_categories['emotion_category_index'] = [emotion_category_indices[c] for c in emotion_categories['emotion_category_id'].tolist()]

# Compute emotion indices and build the whole emotion table
emotions.columns = [u'emotion_id', u'emotion_category_id', u'emotion_name', u'enabled']
all_emotions = sorted(emotions.emotion_id.unique())
emotion_indices = {e: i for i, e in enumerate(all_emotions)}
emotions['emotion_index'] = [emotion_indices[c] for c in emotions['emotion_id'].tolist()]
emotions['plain_name'] = [re.match('[a-zA-z\ ]+\Z', name) is not None for name in emotions.emotion_name]
emotions_full = emotions.merge(emotion_categories, on='emotion_category_id')
emotions_full.columns = emotions_full.columns.astype(str)
emotions_full['emotions_label'] = [emotion_names[name] for name in emotions_full.emotion_name]
emotions_full['emotion_categories_label'] = [category_names[name] for name in emotions_full.emotion_category_name]
emotions_full.to_parquet('../preprocessed/emotions.parquet')

# Find all the users and replace them in the graph
all_users = sorted(set(graph.source_user_id.unique().tolist() + graph.dest_user_id.unique().tolist()))
users = {v: i for i, v in enumerate(all_users)}
users_df = pd.DataFrame([(i, v) for (v, i) in users.items()], columns=['user_index', 'user_id'])
users_df.to_parquet('../preprocessed/users.parquet')
graph['source_user_index'] = [users[u] for u in graph.source_user_id.tolist()]
graph['dest_user_index'] = [users[u] for u in graph.dest_user_id.tolist()]
del graph['source_user_id']
del graph['dest_user_id']
graph.to_csv('../preprocessed/vent-user-indices.edgelist', sep='\t', index=False)

# Compute the complete vent dataset, preprocessing fields
vents_with_users = vents.merge(users_df, on='user_id')
vents_with_users['created_at'] = pd.to_datetime(vents_with_users['created_at'])
del vents_with_users['user_id']
vents_with_users['emotion_index'] = [emotion_indices.get(e, None) for e in vents_with_users['emotion_id'].tolist()]
del vents_with_users['emotion_id']

# Add the emotion category labels directly, so the dataset contains everything at a glance
emotion_cats = emotions_full[['emotion_index', 'emotions_label', 'emotion_category_id', 'emotion_categories_label']]
vents_with_users = vents_with_users.merge(emotion_cats, on='emotion_index', how='inner')
vents_with_users = vents_with_users[vents_with_users.emotion_index != None]
vents_with_users.columns = vents_with_users.columns.astype(str)
vents_with_users.to_parquet('../preprocessed/vent.parquet')

# Compute the robust version of Vent, keeping only the vents that have samples after 2016-07 inclusive
vents_with_users['month'] = vents_with_users.created_at.dt.strftime('%Y-%m')
vents_with_users = vents_with_users[vents_with_users.month >= '2016-07']

# Compute the robust emotion ids in the time range, e.g. having at least 1 sample per month
month_emotion_counts = vents_with_users.groupby(['month', 'emotions_label']).size()
all_months = vents_with_users.month.unique()
robust_emotions = month_emotion_counts.groupby('emotions_label').size()
robust_emotions = set(robust_emotions[robust_emotions == len(all_months)].reset_index().emotions_label.unique())
robust_emotion_indices = {emo: i for i, emo in enumerate(sorted(robust_emotions))}

# Save as the robust dump of emotions
vents_with_users = vents_with_users[vents_with_users.emotions_label.isin(robust_emotions)]
vents_with_users.emotion_index = [robust_emotion_indices[label] for label in vents_with_users.emotions_label]
vents_with_users.to_parquet('../preprocessed/vent-robust.parquet')

