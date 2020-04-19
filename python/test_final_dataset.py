import pandas
import pprint as pp
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from python.ConfigUser import path_processedarticles
from python.ProcessingFunctions import MakeListInLists
from python.AnalysisFunctions_new2 import EstimateLDA, Load_SePL, GetTopicsOfDoc, GetDomTopicOfDoc, MakeCandidates, ReadSePLSentiments, ProcessSentimentScores, ProcessSePLphrases, GetSentimentScores_long, ProcessforSentiment_long, SentenceTokenizer
from python.ProcessingFunctions import SentenceTokenizer

POStag_type = 'NN'

#load df_articles_long2 form sentences processing (line 118) - or save it as dataset and load in again

#Todo: problems with load sepl function
df_sepl = Load_SePL()



df_long_articles2 = pandas.read_csv(path_processedarticles + 'sentences_for_sentiment_l.csv', sep='\t')

#ToDo: check all three dataframes for lda

######################################

POStag_type = 'NN'
#nlp2 = spacy.load('de_core_news_md', disable=['ner', 'parser'])


###########LDA ESTIMATION#####################

#LDA on sentence level

# Read in output file from PreprocessingSentences.py
df_lda_sentence = pandas.read_csv(path_processedarticles + 'csv/sentences_for_lda_{}_l.csv'.format(POStag_type), sep='\t')

lda_sentence = EstimateLDA(df_lda_sentence['sentences_{}_for_lda'.format(POStag_type)])

lda_model = lda_sentence[0]
docsforlda = lda_sentence[1]
dict_lda = lda_sentence[2]
corpus_lda = lda_sentence[3]

# Estimate Topic distribution of sentiment sentences and append to long file
# Todo: pandas df does not lead - not all arguments converted during string formatting
df_long_articles2['Dom_topic_lda_sentence'] = df_long_articles2['sentences_for_sentiment'].apply(lambda x: GetDomTopicOfDoc(x,lda_model=lda_model, dict_lda=dict_lda))

#####

#LDA on paragraph level
df_lda_paragraph = pandas.read_csv(path_processedarticles + 'csv/paragraphs_for_lda_{}_l.csv'.format(POStag_type), sep='\t')

lda_paragraph = EstimateLDA(df_lda_paragraph['paragraphs_{}_for_lda'.format(POStag_type)])
lda_model = lda_paragraph[0]
docsforlda = lda_paragraph[1]
dict_lda = lda_paragraph[2]
corpus_lda = lda_paragraph[3]

df_long_articles2['Dom_topic_lda_paragraph'] = df_long_articles2['sentences_for_sentiment'].apply(lambda x: GetDomTopicOfDoc(x,lda_model=lda_model, dict_lda=dict_lda))


#LDA on article level
df_lda_article = pandas.read_csv(path_processedarticles + 'csv/articles_for_lda_{}_l.csv'.format(POStag_type), sep='\t')

lda_article = EstimateLDA(df_lda_article['articles_{}_for_lda'.format(POStag_type)])
lda_model = lda_article[0]
docsforlda = lda_article[1]
dict_lda = lda_article[2]
corpus_lda = lda_article[3]

df_long_articles2['Dom_topic_lda_article'] = df_long_articles2['sentences_for_sentiment'].apply(lambda x: GetDomTopicOfDoc(x,lda_model=lda_model, dict_lda=dict_lda))


##########Sentiment Analysis




df_long_articles2['Sentiment'] = df_long_articles2['sentences_for_sentiment'].apply(lambda x: GetSentimentScores_long(sent=x, df_sepl=df_sepl))
# Todo: Pandas dataframe does not load - only insert sentiment scores from GetSentimentScores - leave out SePl phares (problem with tuple structure??), RuntimeWarning: Mean of empty slice.



df_long_articles2.to_excel(path_processedarticles + 'test_final_dataset_{}_l.xlsx'.format(POStag_type))

##########################################


