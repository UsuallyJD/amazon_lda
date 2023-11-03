'''
helper functions for review_analysis core file
'''
import re
import numpy as np
import matplotlib.pyplot as plt

from wordcloud import WordCloud

def clean_data(text):
    '''
    cleans data to remove unwanted characters and punctuation.

    parameters:
        text: string

    returns:
        text: modified string
    '''
    text.replace('\\n', ' ')
    text = re.sub(r'[ ]{2, }', ' ', text)
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text.lower().lstrip().strip()

    return text


def get_doc_topic(lda_model, corpus):
    '''
    passes bag-of-words vector into trained LDA model to get topic id of document

    parameters:
    lda_model: Gensim object (must be trained model)
    corpus: matrix of tuples (ie list of lists)

    returns:
        topic_id_list: list
    '''

    # create list for most likely topic ids for each document
    doc_topic_ids = []

    # iterature through the bag-of-words vectors for each doc
    for doc_bow in corpus:

        # create lists for topic ids and probabilities
        topic_ids = []
        topic_probs = []

        # each tuple has a topic id and the prob that the doc belongs to that topic
        topic_id_prob_tuples = lda_model.get_document_topics(doc_bow)

        # iterate through the topic id/prob pairs
        for topic_id_prob in topic_id_prob_tuples:

            # index for topic id
            topic_id = topic_id_prob[0]
            # index for prob that doc belongs that the corresponding topic
            topic_prob = topic_id_prob[1]

            # store all topic ids for doc
            topic_ids.append(topic_id)
            # store all topic probs for doc
            topic_probs.append(topic_prob)

        # get index for topic with highest probability & associated topic id
        likely_topic_id = topic_ids[np.argmax(topic_probs)]
        doc_topic_ids.append(likely_topic_id)

    return doc_topic_ids

def generate_word_cloud(topic_model, topic_id, topic_names, num_words=20):
    '''
    generates word cloud for topic with top (num_words) words most specific to select topic

    parameters:
        topic_model: Gensim object (must be trained model)
        topic_id: int
        topic_names: dictionary of topic names
        num_words: int (default 20)
    
    returns:
        no return (displays WordCloud plot)
    '''

    # get topic-term distribution
    topic_terms = topic_model.get_topic_terms(topic_id, topn=topic_model.num_terms)
    
    # sort terms by topic-specific probability (descending)
    sorted_terms = sorted(topic_terms, key=lambda x: x[1], reverse=True)
    
    # extract top words
    words = [(topic_model.id2word[term_id], prob) for term_id, prob in sorted_terms[:num_words]]
    
    # Generate the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(words))
    
    # plot with presumed topic names
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"Exclusive Word Cloud for {topic_names[topic_id]}")
    plt.show()
