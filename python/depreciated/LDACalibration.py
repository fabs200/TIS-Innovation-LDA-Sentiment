from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
import pandas
import pprint as pp
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from python.ConfigUser import path_processedarticles
from python._ProcessingFunctions import MakeListInLists
from python._AnalysisFunctions import MakeTopicsBOW, LDAHellinger
from python._AnalysisFunctions import EstimateLDA, LDAJaccard, LDACoherence, LDACalibration
from gensim.matutils import jaccard, hellinger
from gensim.models.coherencemodel import CoherenceModel

# Read in output file from PreprocessingSentences.py
df_articles = pandas.read_csv(path_processedarticles + 'csv/sentences_for_lda_analysis_l_short.csv', sep='\t')


test = LDACalibration(topics_start=1, topics_limit=5, topics_step=1, dataframecolumn=df_articles['Article_sentence_nouns_cleaned'], topn=10, num_words=10, metric='coherence', verbose=True)
