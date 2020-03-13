import pandas
import pprint as pp
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from python.ConfigUser import path_processedarticles
import python.main
from python.ProcessingFunctions import MakeListInLists
from nltk.tokenize import word_tokenize

# Read in file with articles from R-Skript ProcessNexisArticles.R
df_articles_lda = pandas.read_csv(path_processedarticles + 'articles_for_lda_analysis.csv', sep='\t')

# Read n file with textbody from R-Skript ProcessNexisArticles.R
df_textbody = pandas.read_csv(path_processedarticles + 'textbody_for_lda_analysis.csv', sep='\t')

# Extract list from dataframe
nouns = MakeListInLists(df_articles_lda['Nouns_lemma'])

# Create a dictionary representation of the documents
dict_nouns = Dictionary(nouns)

# Display
# pp.pprint(dict_nouns.token2id)

# Filter out words that occur less than 20 documents, or more than 50% of the documents
dict_nouns.filter_extremes(no_below=20, no_above=0.2)

# Bag-of-words representation of the documents
corpus_nouns = [dict_nouns.doc2bow(doc) for doc in nouns]

# Make a index to word dictionary
temp = dict_nouns[0]  # This is only to "load" the dictionary
id2word_nouns = dict_nouns.id2token

# Display
pp.pprint(id2word_nouns)

# Display results of Corpus
print(corpus_nouns)
print('Number of unique tokens: {}'.format(len(dict_nouns)))
print('Number of documents: {}'.format(len(corpus_nouns)))

# TODO: save corpus and dictionary to disk and load them back
# save to path_lda_data

lda_nouns = LdaModel(corpus=corpus_nouns, id2word=id2word_nouns, num_topics=5, iterations=300, eval_every=1)

lda_nouns.print_topics(-1)

# Print the Keyword in the 10 topics
pp.pprint(lda_nouns.print_topics())

########################
########################

# Get topic distribution and dominant topic for each document

# Flatten the textbody from the articles to list of lists
textbody = MakeListInLists(df_textbody['Article'])
# TODO: local lda, also load similiar to df_textbody['Article'] a sentence-wise list

#tokenize textbody
textbody_tok = []
for article in textbody:
    for word in article:
        token = word_tokenize(word)
        textbody_tok.append(token)


# Create BOW representation of textbody to use as input for the LDA model
textbody_bow = [dict_nouns.doc2bow(text) for text in textbody_tok]

# Get topic distribution for each document and collect in list of lists
topic_list = []
    for i in range(len(textbody_bow)):
        for word in sublist:
            topic_prob = lda_nouns.get_document_topics(textbody_bow[i])
            topic_list.append(topic_prob)



# Get the dominant topic for each document and collect in list of lists
topic_dom_list = []
    for i in range(len(textbody_bow)):
        topic_dom_list.append([])
        for word in sublist:
            topic_prob = lda_nouns.get_document_topics(textbody_bow[i])
            print(topic_prob)
            topic_dom = max(topic_prob, key=lambda item: item[1])
            print(topic_dom)
            topic_dom_list[-1].append(topic_dom)


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

