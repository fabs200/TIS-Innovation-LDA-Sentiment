import pandas
import pprint as pp
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from python.ConfigUser import path_processedarticles
from nltk.tokenize import word_tokenize
from python.ProcessingFunctions import MakeListInLists
from python.ProcessingFunctions import SentenceCleaner

# Read in file with articles from R-Skript ProcessNexisArticles.R
df_articles_sentences_lda = pandas.read_csv(path_processedarticles + 'csv/sentences_for_lda_analysis.csv', sep='\t',)

# Read in list in list (=1 sentences 1 doc)
sentences_nouns = MakeListInLists(df_articles_sentences_lda['Article_sentence_nouns_cleaned'])

#Todo: noun list for lda on article and paragraph level - as function parameters?

# Read n file with textbody from R-Skript ProcessNexisArticles.R
#df_sentences = pandas.read_csv(path_processedarticles + 'csv/sentences_for_lda_analysis.csv', sep='\t')

# Extract list from dataframe
#nouns = MakeListInLists(df_articles_lda['Nouns_lemma'])

# Create a dictionary representation of the documents
dict_nouns = Dictionary(sentences_nouns)

# Display
# pp.pprint(dict_nouns.token2id)

# Filter out words that occur less than 20 documents, or more than 50% of the documents
# Todo: include filter settings as function parameters
dict_nouns.filter_extremes(no_below=20, no_above=0.5)

# Bag-of-words representation of the documents
corpus_nouns = [dict_nouns.doc2bow(doc) for doc in sentences_nouns]

# Make a index to word dictionary
temp = dict_nouns[0]  # This is only to "load" the dictionary
id2word_nouns = dict_nouns.id2token

# Display
pp.pprint(id2word_nouns)

# Display results of Corpus
#print(corpus_nouns)
print('Number of unique tokens: {}'.format(len(dict_nouns)))
print('Number of documents: {}'.format(len(corpus_nouns)))

# TODO: save corpus and dictionary to disk and load them back
# save to path_lda_data

#todo lda parameters in function
lda_nouns = LdaModel(corpus=corpus_nouns, id2word=id2word_nouns, num_topics=5, iterations=300, eval_every=1)

lda_nouns.print_topics(-1)

# Print the Keyword in the 10 topics
pp.pprint(lda_nouns.print_topics())

########################
########################
# todo two Function: get topic distribution/ get dominant topic
##Todo: make function to apply on dataframe - input one tokenized sentence sentence,

df_articles_sentences_lda['Article_sentiment_sentences_token'] = df_articles_sentences_lda['Article_sentiment_sentences'].apply(lambda x: Sentence(x))
# Read in list in list - sentences for sentiment analysis
sentences_senti = df_articles_sentences_lda['Article_sentiment_sentences'].tolist()
test = SentenceCleaner(sentences_senti)

for x in sentences_senti:
    for sentence in x:
        print (sentence)

sentences_senti2 = df_articles_sentences_lda['Article_sentiment_sentences'].to_list()

article = sentences_senti2[0]

article_new = article.replace('[', '').replace(']', '').replace('\'', '')
article_new = article_new.split(',')

sentences_senti_tokens = []
for word in article_new:
    token = word_tokenize(word)
    sentences_senti_tokens.append(token)

print(sentences_senti_tokens[60])

# Create BOW representation of textbody to use as input for the LDA model
sentences_bow = [dict_nouns.doc2bow(text) for text in sentences_senti_tokens]

# Get topic distribution for each document and collect in list of lists
topic_list = []
for i in range(0, len(sentences_bow)):
    topic_prob = lda_nouns.get_document_topics(sentences_bow[1])
    topic_list.append(topic_prob)

topic_list[61]


# Get the dominant topic for each document and collect in list of lists
topic_dom_list = []
for i in range(len(sentences_bow)):
    topic_dom_list.append([])
    topic_prob = lda_nouns.get_document_topics(sentences_bow[i])
    print(topic_prob)
    topic_dom = max(topic_prob, key=lambda item: item[1])
    print(topic_dom)
    topic_dom_list[-1].append(topic_dom)

topic_list[60]
topic_dom_list[60]

# Save to help df
df_help_noun_lemma_list = pandas.DataFrame({'x': noun_lemma_list})
# Save to help df
df_help_topic_list = pandas.DataFrame({'x': topic_list})
df_help_topic_dom_list = pandas.DataFrame({'x': topic_dom_list})

# Create id increasing (needed to merge to original df)
df_help_topic_list.insert(0, 'ID_incr', range(1, 1 + len(df_help_topic_list)))
df_help_topic_dom_list.insert(0, 'ID_incr', range(1, 1 + len(df_help_topic_dom_list)))

# Merge
df_textbody = (df_textbody.merge(df_help_topic_list, left_on='ID_incr', right_on='ID_incr')).rename(columns={'x': 'Topic_distribution'})

df_textbody = (df_textbody.merge(df_help_topic_dom_list, left_on='ID_incr', right_on='ID_incr')).rename(columns={'x': 'Topic_dominant'})


