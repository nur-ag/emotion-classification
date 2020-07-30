import re
import numpy as np
import pandas as pd

emotions = pd.read_csv('../data/emotions.csv')
emotion_categories = pd.read_csv('../data/emotion_categories.csv')
graph = pd.read_csv('../data/vent.edgelist', names=['source_user_id', 'dest_user_id'], sep=' ')
vents = pd.read_csv('../data/vents.csv')

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
vents_with_users['emotion_index'] = [emotion_indices[e] for e in vents_with_users['emotion_id'].tolist()]
del vents_with_users['emotion_id']

# Add the emotion category indices directly, so the dataset contains everything at a glance
emotion_cats = emotions_full[['emotion_index', 'emotion_category_index', 'emotion_name', 'enabled', 'plain_name']]
vents_with_users = vents_with_users.merge(emotion_cats, on='emotion_index')
vents_with_users.columns = vents_with_users.columns.astype(str)
vents_with_users.to_parquet('../preprocessed/vent.parquet')
