import os
import pandas

from python.ConfigUser import path_data, path_project
from python._HelpFunctions import filter_sentiment_params
from python.params import params as p

"""
------------------------------------------
08_descriptive_tables
------------------------------------------

"""

# unpack POStag type, lda_levels to run lda on, lda_level to get domtopic from
POStag, lda_level_fit, sent = p['POStag'], p['lda_level_fit'], p['sentiment_list']
if 'article' in p['lda_level_domtopic']: lda_level_domtopic = 'article'

# create folder in graphs with currmodel
os.makedirs(path_project + "tables/{}/model_{}".format(sent, p['currmodel']), exist_ok=True)
# create folder in currmodel with specified lda_level_domtopic
os.makedirs(path_project + "tables/{}/model_{}/{}".format(sent, p['currmodel'], lda_level_domtopic), exist_ok=True)

# Load long file (sentence-level)
print('Loading lda_results_{}_l.csv'.format(p['currmodel']))
df_long = pandas.read_csv(path_data + 'csv/lda_results_{}_l.csv'.format(p['currmodel']), sep='\t', na_filter=False)

# Rename target sentiment variable, make numeric
df_long = df_long.rename(columns={'ss_{}_mean'.format(sent): 'sentiscore_mean'})
df_long['sentiscore_mean'] = pandas.to_numeric(df_long['sentiscore_mean'], errors='coerce')
df_long['DomTopic_arti_arti_prob'] = pandas.to_numeric(df_long['DomTopic_arti_arti_prob'], errors='coerce')

# calculate average sentiment per article and merge to df_long
df_long = pandas.merge(df_long.drop('sentiscore_mean', axis=1),
                       df_long.groupby('Art_ID', as_index=False).sentiscore_mean.agg(['count', 'min', 'max', 'mean', 'std']),
                       how='left', on=['Art_ID'])

# Filter df_long
df_long = filter_sentiment_params(df_long, df_sentiment_list=sent)

# Select articles and columns
df_long = df_long[df_long['Art_unique'] == 1][['DomTopic_arti_arti_id',
                                               'year', 'quarter', 'month',
                                               'Newspaper', 'count', 'min', 'max', 'mean', 'std', 'DomTopic_arti_arti_prob',
                                               'articles_text']]

# convert dtypes
df_long['month'] = pandas.to_datetime(df_long['month'], format='%Y-%m')
for stat in ['min', 'max', 'mean', 'std', 'DomTopic_arti_arti_prob']:
    df_long[stat] = pandas.to_numeric(df_long[stat], errors='coerce')
    df_long[stat] = pandas.to_numeric(df_long[stat], errors='coerce')

# select date range
df_long = df_long[(df_long['month'] >= '2009-1-1')]
df_long = df_long[(df_long['month'] <= '2020-1-1')]

# replace everything in brackets from Newspaper
df_long['Newspaper'] = df_long.Newspaper.replace(to_replace='\([^)]*\)', value='', regex=True).str.strip()

"""
###################### Table 1: descriptives, by topic & year ######################
"""

# group by topics, calculate mean, sd, and reshape long to wide
df_wide_bytopics_mean = df_long.groupby(['DomTopic_arti_arti_id', 'month'])[['mean']].mean().reset_index().pivot(
    index='month', columns='DomTopic_arti_arti_id', values='mean')
df_wide_bytopics_std = df_long.groupby(['DomTopic_arti_arti_id', 'month'])[['std']].mean().reset_index().pivot(
    index='month', columns='DomTopic_arti_arti_id', values='std')
df_wide_bytopics_prob = df_long.groupby(['DomTopic_arti_arti_id', 'month'])[['DomTopic_arti_arti_prob']].mean()\
    .reset_index().pivot(index='month', columns='DomTopic_arti_arti_id', values='DomTopic_arti_arti_prob')\
    .rename(columns={'DomTopic_arti_arti_prob': 'prob'})
df_wide_bytopics_count = df_long.groupby(['DomTopic_arti_arti_id', 'month'])[['count']].mean()\
    .reset_index().pivot(index='month', columns='DomTopic_arti_arti_id', values='count')\
    .rename(columns={'DomTopic_arti_arti_prob': 'count'})
df_wide_bytopics_min = df_long.groupby(['DomTopic_arti_arti_id', 'month'])[['min']].mean()\
    .reset_index().pivot(index='month', columns='DomTopic_arti_arti_id', values='min')\
    .rename(columns={'DomTopic_arti_arti_prob': 'min'})
df_wide_bytopics_max = df_long.groupby(['DomTopic_arti_arti_id', 'month'])[['max']].mean()\
    .reset_index().pivot(index='month', columns='DomTopic_arti_arti_id', values='max')\
    .rename(columns={'DomTopic_arti_arti_prob': 'max'})

# aggregate by year
df_aggr_bytopics_mean_y = df_wide_bytopics_mean.groupby(pandas.Grouper(freq='Y')).mean().reset_index().rename(columns={'month': 'year'})
df_aggr_bytopics_std_y = df_wide_bytopics_std.groupby(pandas.Grouper(freq='Y')).mean().reset_index().rename(columns={'month': 'year'})
df_aggr_bytopics_prob_y = df_wide_bytopics_prob.groupby(pandas.Grouper(freq='Y')).mean().reset_index().rename(columns={'month': 'year'})
df_aggr_bytopics_count_y = df_wide_bytopics_count.groupby(pandas.Grouper(freq='Y')).mean().reset_index().rename(columns={'month': 'year'})
df_aggr_bytopics_min_y = df_wide_bytopics_min.groupby(pandas.Grouper(freq='Y')).mean().reset_index().rename(columns={'month': 'year'})
df_aggr_bytopics_max_y = df_wide_bytopics_max.groupby(pandas.Grouper(freq='Y')).mean().reset_index().rename(columns={'month': 'year'})

### Reformat dates
df_aggr_bytopics_mean_y['year'] = pandas.DatetimeIndex(df_aggr_bytopics_mean_y.iloc[:, 0]).year
df_aggr_bytopics_std_y['year'] = pandas.DatetimeIndex(df_aggr_bytopics_std_y.iloc[:, 0]).year
df_aggr_bytopics_prob_y['year'] = pandas.DatetimeIndex(df_aggr_bytopics_prob_y.iloc[:, 0]).year
df_aggr_bytopics_count_y['year'] = pandas.DatetimeIndex(df_aggr_bytopics_count_y.iloc[:, 0]).year
df_aggr_bytopics_min_y['year'] = pandas.DatetimeIndex(df_aggr_bytopics_min_y.iloc[:, 0]).year
df_aggr_bytopics_max_y['year'] = pandas.DatetimeIndex(df_aggr_bytopics_max_y.iloc[:, 0]).year


# before appending all dfs, define a help-id to sort properly later
df_aggr_bytopics_count_y['help_id_'] = 1
df_aggr_bytopics_mean_y['help_id_'] = 2
df_aggr_bytopics_std_y['help_id_'] = 3
df_aggr_bytopics_min_y['help_id_'] = 4
df_aggr_bytopics_max_y['help_id_'] = 5
df_aggr_bytopics_prob_y['help_id_'] = 6

# append all dfs and sort
df_agg_bytopics = df_aggr_bytopics_count_y.append([df_aggr_bytopics_mean_y,
                                                   df_aggr_bytopics_std_y,
                                                   df_aggr_bytopics_min_y,
                                                   df_aggr_bytopics_max_y,
                                                   df_aggr_bytopics_prob_y], ignore_index=True)\
    .sort_values(by=['year', 'help_id_'])\
    .drop(columns=['help_id_'])
# insert a list with stats as first column
stats_col = ['n', 'mean', 'sd', 'min', 'max', 'probability']*int(df_agg_bytopics.shape[0]/6)
df_agg_bytopics.insert(0, 'stats', stats_col)
df_agg_bytopics = df_agg_bytopics.set_index(['stats', 'year'])

# export as excel
df_agg_bytopics.to_excel(path_project + "tables/{}/model_{}/01_descriptive_sentiment.xlsx".format(sent, p['currmodel']))

"""
###################### Table 2: descriptives of sentiment and article count by year ######################
"""

# group by year, calculate mean, sd, min, max, count articles
df_articles_by_year = df_long.groupby('year').agg({'count': 'count',
                                                   'mean': 'mean',
                                                   'min': 'min',
                                                   'max': 'max'})

# export as excel
df_articles_by_year.to_excel(path_project + "tables/{}/model_{}/02_descriptive_sentiment.xlsx".format(sent, p['currmodel']))



print('done')

###
