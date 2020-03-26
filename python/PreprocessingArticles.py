import pandas
import re
from nltk.corpus import stopwords
import spacy
from spacy.pipeline import SentenceSegmenter
from germalemma import GermaLemma
from python.ConfigUser import path_processedarticles
from python.ProcessingFunctions import Sentencizer
# from textblob import NLTKPunktTokenizer

# Read in file with articles from R-Skript ProcessNexisArticles.R
df_articles = pandas.read_feather(path_processedarticles + 'autofiles_withbattery.feather')

# convert all words to lower case
df_articles['Article'] = [i.lower() for i in df_articles['Article']]

# Drop duplicates
df_articles.drop_duplicates(subset=['Article', 'Date'], inplace=True)
df_articles.drop_duplicates(subset=['Headline'], inplace=True)

# TODO: drop duplicates Articles based on similaritiy index

# Remove text which defines end of articles
df_articles['Article'] = df_articles['Article'].str.split('graphic').str[0]
df_articles['Article'] = df_articles['Article'].str.split('classification language').str[0]
cutpage = re.compile(r'(kommentar seite \d+)')
cutpage2 = re.compile(r'deliverynotification')
df_articles['Article'] = [cutpage.sub(lambda m: (m.group(1) if m.group(1) else " "), x) for x in df_articles['Article'].tolist()]
df_articles['Article'] = [cutpage2.sub(lambda m: (m.group(1) if m.group(1) else " "), x) for x in df_articles['Article'].tolist()]

# Make Backup
df_articles['Article_backup'] = df_articles['Article']

# Create id increasing (needed to merge help files later)
df_articles.insert(0, 'ID_incr', range(1, 1 + len(df_articles)))

# Remove all numbers
df_articles['Article'] = df_articles['Article'].str.replace('\d+', '')

# Remove additional words and words of length 1
drop_words = ['www', 'dpa', 'de', 'foto', 'webseite', 'herr', 'vdi', 'interview']
df_articles['Article'] = df_articles['Article'].apply(lambda x: " ".join(x for x in x.split() if x not in drop_words))
df_articles['Article'] = df_articles['Article'].apply(lambda x: re.sub(r'(^|\s+)(\S(\s+|$))', ' ', x))

# Remove punctuation except hyphen and apostrophe between words
p = re.compile(r"(\b[-']\b)|[\W_]")
df_articles['Article'] = [p.sub(lambda m: (m.group(1) if m.group(1) else " "), x) for x in df_articles['Article'].tolist()]

# Download list of stopwords from nltk (needed to be done once)
# nltk.download('stopwords')

# Load German stop words and apply to articles
stop = stopwords.words('german')
df_articles['Article'] = df_articles['Article'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
#TODO: check for other stop-words list - spacy, solariz github (check if negation is removed - sentiment analysis)

# POS tagging (time-consuming!)
#TODO: maybe use faster POS-tagging, e.g. NLTK tagger or ClassifierBasedGermanTagger using TIGER corpus, but spacy has higher accuracy
nlp = spacy.load('de_core_news_md', disable=['ner', 'parser'])
df_articles['Article_POS'] = df_articles['Article'].apply(lambda x: nlp(x))

# Create new column including only nouns (all noun types from STTS tagset)
df_articles['Nouns'] = df_articles['Article_POS'].apply(lambda x: [token for token in x if token.tag_.startswith('NN')])

# remove words with length==1
df_articles['Nouns'] = df_articles['Nouns'].apply(lambda x: [word for word in x if len(x)>1])
# df_articles['Nounverbs'] = df_articles['Nounverbs'].apply(lambda x: [word for word in x if len(x)>1])

# Lemmatization
lemmatizer = GermaLemma()

# Lemmatization of Nouns
noun_list = df_articles['Nouns'].tolist()

global noun_lemma_list
noun_lemma_list = []
for doc in noun_list:
    noun_lemma_list.append([])
    for token in doc:
        token_lemma = lemmatizer.find_lemma(token.text, token.tag_)
        token_lemma = token_lemma.lower()
        noun_lemma_list[-1].append(token_lemma)

# Save to help df
df_help_noun_lemma_list = pandas.DataFrame({'x': noun_lemma_list})

# Create id increasing (needed to merge to original df)
df_help_noun_lemma_list.insert(0, 'ID_incr', range(1, 1 + len(df_help_noun_lemma_list)))

# Merge df_help_noun_lemma_list to df_help_noun_lemma_list and rename
df_articles = (df_articles.merge(df_help_noun_lemma_list, left_on='ID_incr', right_on='ID_incr')).rename(columns={'x': 'Nouns_lemma'})

# Export data to excel
df_articles.to_excel(path_processedarticles + 'articles_for_lda_analysis.xlsx')

# Export data to csv (will be read in again in LDACalibration.py)
df_articles_export = df_articles[['ID_incr', 'ID', 'Date', 'Nouns', 'Nouns_lemma']]
df_articles_export.to_csv(path_processedarticles + 'articles_for_lda_analysis.csv', sep='\t', index=False)

#Export textbody data to csv (for aspect extraction)
df_textbody_export = df_articles[['ID_incr', 'ID', 'Date','Article']]
df_textbody_export.to_csv(path_processedarticles + 'textbody_for_lda_analysis.csv', sep='\t', index=False)

# Clean up to keep RAM small
del df_articles, df_help_noun_lemma_list, df_articles_export
del stop, stopwords

###
