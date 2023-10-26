import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import spacy
spacy.util.fix_random_seed(0)

import pyLDAvis
import pyLDAvis.gensim_models

import gensim.corpora as corpora

# help functions
from help_functions import clean_data, \
                           compute_coherence_values, \
                           plot_coherence_v_no_topics, \
                           get_topic_id_lookup_dict, \
                           get_topic_ids_for_docs

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('reviews_data.csv')

# create a clean_text column by applying  clean_data to your text
df['clean_text'] = df['reviews.text'].apply(clean_data)

# create mask for docs in Electronics category
electronics_mask = df.primaryCategories.isin(["Electronics"])
df_electronics = df[electronics_mask]

# load in the spaCy language model
spacy.cli.download('en_core_web_sm')
nlp = spacy.load('en_core_web_sm')

# create lemma tokens
df_electronics['lemmas'] = df_electronics['clean_text'].apply(
    lambda x: [token.lemma_ for token in nlp(x) if (token.is_stop != True) and (token.is_punct != True)])

# Create lemma dictionary using Dictionary
id2word = corpora.Dictionary(df_electronics['lemmas'])
id2word.filter_extremes(no_below=3, no_above=0.5)

# Create Term Document Frequency list
corpus = [id2word.doc2bow(lemma) for lemma in df_electronics['lemmas']]

model_list, coherence_values = compute_coherence_values(dictionary=id2word,
                                                        corpus=corpus,
                                                        id2word=id2word,
                                                        texts=df_electronics['lemmas'],
                                                        start=2, limit=16, step=2)

x = range(start=2, limit=16, step=2)

# graph coherence score v. no. topics to determine how many topics to generate
plot_coherence_v_no_topics(x, coherence_values)

max_coherence_val_index = np.argmax(coherence_values)
lda_trained_model = model_list[max_coherence_val_index]

# plot topics using pyLDAvis
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim_models.prepare(lda_trained_model, corpus, id2word)
vis

# create a dictionary
# keys - use topic ids from pyLDAvis visualization
# values - topic names that you create
# save dictionary to `vis_topic_name_dict`

vis_topic_name_dict = {
                        1: 'primary_user',
                        2: 'gift_pruchase'
                        }

topic_name_dict = get_topic_id_lookup_dict(vis, vis_topic_name_dict)

# use get_topic_ids_for_docs to get the topic id for each doc in the corpus
doc_topic_ids = get_topic_ids_for_docs(lda_trained_model, corpus)

# create new topic_id feature in df_electronics
df_electronics['topic_id'] = doc_topic_ids

# iterate through topic_id and use the lookup dict `topic_name_dict` to assign each document a topic name
df_electronics['new_topic_name'] = df_electronics['topic_id'].apply(lambda topic_id: topic_name_dict[topic_id])

cols = ["reviews.text", "new_topic_name", "topic_id"]
df_electronics[cols].head(15)
