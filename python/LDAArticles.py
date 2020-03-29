import pandas
import pprint as pp
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from python.ConfigUser import path_processedarticles
from python.ProcessingFunctions import MakeListInLists

# Read in output file from PreprocessingSentences.py
df_articles = pandas.read_csv(path_processedarticles + 'feather/articles_for_lda_analysis.csv', sep='\t')

articles = df_articles['Nouns_lemma'].to_list()

# Read in list in list (=1 sentences 1 doc)
articles = MakeListInLists(articles)

# Create a dictionary representation of the documents
dict_nouns = Dictionary(articles)

# Display
# pp.pprint(dict_nouns.token2id)

# Filter out words that occur less than 20 documents, or more than 50% of the documents
dict_nouns.filter_extremes(no_below=4, no_above=0.4)

# Bag-of-words representation of the documents
corpus_nouns = [dict_nouns.doc2bow(doc) for doc in articles]

# Make a index to word dictionary
temp = dict_nouns[0]  # This is only to "load" the dictionary
id2word_nouns = dict_nouns.id2token

# Display
#pp.pprint(id2word_nouns)

# Display results of Corpus
# print(corpus_nouns)
# print('Number of unique tokens: {}'.format(len(dict_nouns)))
# print('Number of documents: {}'.format(len(corpus_nouns)))

# TODO: save corpus and dictionary to disk and load them back
# save to path_lda_data

lda_nouns = LdaModel(corpus=corpus_nouns, id2word=id2word_nouns, num_topics=10, iterations=300, eval_every=1)

lda_nouns.print_topics(-1)

# use Jaccard distance (Topic Chains for Understanding a News Corpus)

from gensim.matutils import kullback_leibler, jaccard, hellinger, sparse2full
kullback_leibler(list_bow, 10)

list_bow[0]
jaccard(list_bow[0], list_bow[2])




# Print the Keyword in the 10 topics
pp.pprint(lda_nouns.print_topics())


list = lda_nouns.show_topics()

list_bow = []
for topic in list:
    help = make_topics_bow(topic)
    list_bow.append(help)

for i in list_bow:
    for j in list_bow:
        jac = jaccard(i, j)
        print(jac)
        sum = 0+jac
print(sum)

def make_topics_bow(topic):
    # split on strings to get topics and the probabilities
    topic = topic[1].split('+')
    # list to store topic bows
    topic_bow = []
    for word in topic:
        # split topic probability and word
        prob, word = word.split('*')
        # get rid of spaces
        word = word.replace(" ","").replace('"','')
        # map topic words to dictionary id
        word_id = dict_nouns.doc2bow([word])
        # append word_id and topic probability
        topic_bow.append((word_id[0][0], float(prob)))

    return topic_bow


test2 = make_topics_bow(top1)

########################
########################

#u_mass coherence measure
from gensim.models.coherencemodel import CoherenceModel
lda_nouns_cm = CoherenceModel(model=lda_nouns, corpus=corpus_nouns, dictionary=dict_nouns, coherence="u_mass")
print(lda_nouns_cm.get_coherence())

##we use coherence measure c_v as suggested by Röder et al. 2015, bceause it has the highest correlation with human interpretability

#TODO: test alpha = 'auto' and eta ='auto' in lda model and add to compute_coherence function

##with coherence measure: u_mass

def compute_coherence_values(dictionary, corpus, texts, id2word, topics_limit, topics_start, topics_step):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics
    TODO: add random state to get same results

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(topics_start, topics_limit, topics_step):
        model = LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics, alpha='auto', eta='auto', random_state=203495)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
        print('num_topics:', num_topics, 'coherence:', coherencemodel.get_coherence())
    return model_list, coherence_values

start, limit, step = 1, 10, 1

model_list, coherence_values = compute_coherence_values(dictionary=dict_nouns, id2word=id2word_nouns, corpus=corpus_nouns, texts=sentences, topics_start=start, topics_limit=limit, topics_step=step)
# Show graph
import matplotlib.pyplot as plt
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()
