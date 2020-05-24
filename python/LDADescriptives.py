import pandas, os, time, dtale
from python.ConfigUser import path_data, path_project
from python.params import params as p
import matplotlib.pyplot as plt
import numpy as np

"""
------------------------------------------
LDADescriptives.py - Overview of graphs
------------------------------------------
# - How many Newspapers, how many articles do they publish TODO
# - How many valid sentiments, per year, per topic, per article TODO
# - How many articles per topic TODO
# - How often is a topic occurring per year TODO

"""

# unpack POStag type, lda_levels to run lda on, lda_level to get domtopic from
POStag, lda_level_fit, lda_level_domtopic = p['POStag'], p['lda_level_fit'], p['lda_level_domtopic']

# create folder in graphs with currmodel
os.makedirs(path_project + "graph/model_{}".format(p['currmodel']), exist_ok=True)

# Load long file
print('Loading lda_results_{}_l.csv'.format(p['currmodel']))
df_long = pandas.read_csv(path_data + 'csv/lda_results_{}_l.csv'.format(p['currmodel']), sep='\t', na_filter=False)

# Select articles and columns
df_long = df_long[df_long['Art_unique'] == 1][['DomTopic_arti_arti_id',
                                               'year', 'quarter', 'month',
                                               'Newspaper', 'sentiscore_mean', 'articles_text']]

# convert dtypes
df_long['month'] = pandas.to_datetime(df_long['month'], format='%Y-%m')
df_long['sentiscore_mean'] = pandas.to_numeric(df_long['sentiscore_mean'], errors='coerce')

# select date range
df_long = df_long[(df_long['month'] >= '2007-1-1')]

# replace everything in brackets from Newspaper
df_long['Newspaper'] = df_long.Newspaper.replace(to_replace='\([^)]*\)', value='', regex=True).str.strip()


"""
######################  ######################
"""



