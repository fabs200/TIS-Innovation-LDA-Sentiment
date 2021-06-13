import pandas, os, time, dtale
from python.ConfigUser import path_data, path_project
from python._HelpFunctions import filter_sentiment_params
from python.params import params as p
import warnings
import re

"""
------------------------------------------
09_robustness_check.py
------------------------------------------

### Prepare Data for robustness check
Follow Dehler and draw a sample of 10 articles per topic which have highest probability of  assigned dominant topic
and export them to excel
"""

# Ignore some warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Specify topics
tc1 = 'Industry'
tc2 = 'R&D'
tc3 = 'Infrastructure'
tc4 = 'Usability'
tc5 = 'Policy'
topics = ['topics', tc1, tc2, tc3, tc4, tc5]

# unpack POStag type, lda_levels to run lda on, lda_level to get domtopic from
POStag, lda_level_fit, sent = p['POStag'], p['lda_level_fit'], p['sentiment_list']
if 'article' in p['lda_level_domtopic']: lda_level_domtopic = 'article'

# Load long file (sentence-level)
print('Loading lda_results_{}_l.csv'.format(p['currmodel']))
df_long = pandas.read_csv(path_data + 'csv/lda_results_{}_l.csv'.format(p['currmodel']), sep='\t', na_filter=False)

# Rename target sentiment variable, make numeric
df_long = df_long.rename(columns={'ss_{}_mean'.format(sent): 'sentiscore_mean'})
df_long['sentiscore_mean'] = pandas.to_numeric(df_long['sentiscore_mean'], errors='coerce')
df_long['DomTopic_arti_arti_prob'] = pandas.to_numeric(df_long['DomTopic_arti_arti_prob'], errors='coerce')

# calculate average sentiment per article and merge to df_long
df_long = pandas.merge(df_long.drop('sentiscore_mean', axis=1),
                       df_long.groupby('Art_ID', as_index=False).sentiscore_mean.mean(),
                       how='left', on=['Art_ID'])

# Filter df_long
df_long = filter_sentiment_params(df_long, df_sentiment_list=sent)

# Select articles and columns
df_long = df_long[df_long['Art_unique'] == 1][['DomTopic_arti_arti_id',
                                               'year', 'quarter', 'month',
                                               'Newspaper', 'sentiscore_mean', 'DomTopic_arti_arti_prob', 'articles_text']]

# convert dtypes
df_long['month'] = pandas.to_datetime(df_long['month'], format='%Y-%m')
df_long['sentiscore_mean'] = pandas.to_numeric(df_long['sentiscore_mean'], errors='coerce')
df_long['sentiscore_mean'] = pandas.to_numeric(df_long['sentiscore_mean'], errors='coerce')

# actual number of articles with sentiment score and assigned dominant topic
print('Number of actual articles ({} to {}) with sentiment score and assigned dominant topic: {}'.format(
    df_long.year.min(),
    df_long.year.max(),
    df_long.DomTopic_arti_arti_id.__len__()))

# select date range
df_long = df_long[(df_long['month'] >= '2009-1-1')]
df_long = df_long[(df_long['month'] <= '2020-1-1')]

# actual number of articles with sentiment score and assigned dominant topic
print('Number of actual articles ({} to {}) with sentiment score and assigned dominant topic: {}'.format(
    df_long.year.min(),
    df_long.year.max(),
    df_long.DomTopic_arti_arti_id.__len__()))

# replace everything in brackets from Newspaper
df_long['Newspaper'] = df_long.Newspaper.replace(to_replace='\([^)]*\)', value='', regex=True).str.strip()

# write topic names into cells
df_long['topic'] = None
df_long.loc[df_long['DomTopic_arti_arti_id'] == '0.0', 'topic'] = tc1
df_long.loc[df_long['DomTopic_arti_arti_id'] == '1.0', 'topic'] = tc2
df_long.loc[df_long['DomTopic_arti_arti_id'] == '2.0', 'topic'] = tc3
df_long.loc[df_long['DomTopic_arti_arti_id'] == '3.0', 'topic'] = tc4
df_long.loc[df_long['DomTopic_arti_arti_id'] == '4.0', 'topic'] = tc5

# read in lda topic words
lda_file = open(path_project + "lda/{}/model_{}/topics.txt".format(p['lda_level_fit'][0], p['currmodel']), 'r')
lda_topicwords = lda_file.read()
lda_topicwords = re.findall(r'(\(.*\'\))', lda_topicwords)
lda_topicwords = [t.replace("\'", "") for t in lda_topicwords]

# add topic words as a column article
df_long['topic_words'] = None
df_long.loc[df_long['DomTopic_arti_arti_id'] == '0.0', 'topic_words'] = lda_topicwords[0]
df_long.loc[df_long['DomTopic_arti_arti_id'] == '1.0', 'topic_words'] = lda_topicwords[1]
df_long.loc[df_long['DomTopic_arti_arti_id'] == '2.0', 'topic_words'] = lda_topicwords[2]
df_long.loc[df_long['DomTopic_arti_arti_id'] == '3.0', 'topic_words'] = lda_topicwords[3]
df_long.loc[df_long['DomTopic_arti_arti_id'] == '4.0', 'topic_words'] = lda_topicwords[4]

# extract highest assigned articles and gather in one df
df_sample = df_long[df_long['DomTopic_arti_arti_id']=='0.0'].sort_values('DomTopic_arti_arti_prob', ascending=False).head(10)
df_sample = df_sample.append(df_long[df_long['DomTopic_arti_arti_id']=='1.0'].sort_values('DomTopic_arti_arti_prob', ascending=False).head(10))
df_sample = df_sample.append(df_long[df_long['DomTopic_arti_arti_id']=='2.0'].sort_values('DomTopic_arti_arti_prob', ascending=False).head(10))
df_sample = df_sample.append(df_long[df_long['DomTopic_arti_arti_id']=='3.0'].sort_values('DomTopic_arti_arti_prob', ascending=False).head(10))
df_sample = df_sample.append(df_long[df_long['DomTopic_arti_arti_id']=='4.0'].sort_values('DomTopic_arti_arti_prob', ascending=False).head(10))

# set articles_text as last column in df
df_sample.insert(loc = len(df_sample.columns),
                 column = 'article_text',
                 value = df_sample.articles_text)

# export as excel
if not os.path.exists(path_project + 'data/final/'):
    os.makedirs(path_project + 'data/final/')
df_sample.to_excel(path_project + 'data/final/robustness_check_top10_assigned_articles.xlsx')

###
