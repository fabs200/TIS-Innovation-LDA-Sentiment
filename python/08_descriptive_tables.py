import os
import pandas

from python.ConfigUser import path_data, path_project
from python._HelpFunctions import filter_sentiment_params
from python.params import params as p

"""
------------------------------------------
08_descriptive_tables
------------------------------------------
* Table 1: descriptives, by topic & year
* Table 2: descriptives of sentiment and article count by year
* Table 3: sentiment over time, by topic and by top publishers
* Table 4: descriptives, by year, on full sample AND by topic
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

# actual number of articles with sentiment score and assigned dominant topic
print('Number of actual articles ({} to {}) with sentiment score and assigned dominant topic: {}'.format(
    df_long.year.min(),
    df_long.year.max(),
    df_long.DomTopic_arti_arti_id.__len__()))

# replace everything in brackets from Newspaper
df_long['Newspaper'] = df_long.Newspaper.replace(to_replace='\([^)]*\)', value='', regex=True).str.strip()

# prepare counts
df_long = df_long.rename(columns = {'count': 'sentences_count'})
df_long['count'] = df_long['mean'].notna().astype(int)

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
df_wide_bytopics_count = df_long.groupby(['DomTopic_arti_arti_id', 'month'])[['count']].sum()\
    .reset_index().pivot(index='month', columns='DomTopic_arti_arti_id', values='count')\
    .rename(columns={'DomTopic_arti_arti_prob': 'count'})
df_wide_bytopics_sentencescount = df_long.groupby(['DomTopic_arti_arti_id', 'month'])[['sentences_count']].sum()\
    .reset_index().pivot(index='month', columns='DomTopic_arti_arti_id', values='sentences_count')\
    .rename(columns={'DomTopic_arti_arti_prob': 'sentences_count'})
df_wide_bytopics_min = df_long.groupby(['DomTopic_arti_arti_id', 'month'])[['min']].min()\
    .reset_index().pivot(index='month', columns='DomTopic_arti_arti_id', values='min')\
    .rename(columns={'DomTopic_arti_arti_prob': 'min'})
df_wide_bytopics_max = df_long.groupby(['DomTopic_arti_arti_id', 'month'])[['max']].max()\
    .reset_index().pivot(index='month', columns='DomTopic_arti_arti_id', values='max')\
    .rename(columns={'DomTopic_arti_arti_prob': 'max'})

# aggregate by year
df_aggr_bytopics_mean_y = df_wide_bytopics_mean.groupby(pandas.Grouper(freq='Y')).mean().reset_index().rename(columns={'month': 'year'})
df_aggr_bytopics_std_y = df_wide_bytopics_std.groupby(pandas.Grouper(freq='Y')).mean().reset_index().rename(columns={'month': 'year'})
df_aggr_bytopics_prob_y = df_wide_bytopics_prob.groupby(pandas.Grouper(freq='Y')).mean().reset_index().rename(columns={'month': 'year'})
df_aggr_bytopics_count_y = df_wide_bytopics_count.groupby(pandas.Grouper(freq='Y')).sum().reset_index().rename(columns={'month': 'year'})
df_aggr_bytopics_sentencescount_y = df_wide_bytopics_sentencescount.groupby(pandas.Grouper(freq='Y')).sum().reset_index().rename(columns={'month': 'year'})
df_aggr_bytopics_min_y = df_wide_bytopics_min.groupby(pandas.Grouper(freq='Y')).min().reset_index().rename(columns={'month': 'year'})
df_aggr_bytopics_max_y = df_wide_bytopics_max.groupby(pandas.Grouper(freq='Y')).max().reset_index().rename(columns={'month': 'year'})

### Reformat dates
df_aggr_bytopics_mean_y['year'] = pandas.DatetimeIndex(df_aggr_bytopics_mean_y.iloc[:, 0]).year
df_aggr_bytopics_std_y['year'] = pandas.DatetimeIndex(df_aggr_bytopics_std_y.iloc[:, 0]).year
df_aggr_bytopics_prob_y['year'] = pandas.DatetimeIndex(df_aggr_bytopics_prob_y.iloc[:, 0]).year
df_aggr_bytopics_count_y['year'] = pandas.DatetimeIndex(df_aggr_bytopics_count_y.iloc[:, 0]).year
df_aggr_bytopics_sentencescount_y['year'] = pandas.DatetimeIndex(df_aggr_bytopics_sentencescount_y.iloc[:, 0]).year
df_aggr_bytopics_min_y['year'] = pandas.DatetimeIndex(df_aggr_bytopics_min_y.iloc[:, 0]).year
df_aggr_bytopics_max_y['year'] = pandas.DatetimeIndex(df_aggr_bytopics_max_y.iloc[:, 0]).year


# before appending all dfs, define a help-id to sort properly later
df_aggr_bytopics_count_y['help_id_'] = 1
df_aggr_bytopics_mean_y['help_id_'] = 2
df_aggr_bytopics_std_y['help_id_'] = 3
df_aggr_bytopics_min_y['help_id_'] = 4
df_aggr_bytopics_max_y['help_id_'] = 5
df_aggr_bytopics_prob_y['help_id_'] = 6
df_aggr_bytopics_sentencescount_y['help_id_'] = 7

# append all dfs and sort
df_agg_bytopics = df_aggr_bytopics_count_y.append([df_aggr_bytopics_mean_y,
                                                   df_aggr_bytopics_std_y,
                                                   df_aggr_bytopics_min_y,
                                                   df_aggr_bytopics_max_y,
                                                   df_aggr_bytopics_prob_y,
                                                   df_aggr_bytopics_sentencescount_y], ignore_index=True)\
    .sort_values(by=['year', 'help_id_'])\
    .drop(columns=['help_id_'])
# insert a list with stats as first column
stats_col = ['n', 'mean', 'sd', 'min', 'max', 'probability', 'sentences_count']*int(df_agg_bytopics.shape[0]/7)
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
                                                   'std': 'mean',
                                                   'min': 'min',
                                                   'max': 'max'})

# export as excel
df_articles_by_year.to_excel(path_project + "tables/{}/model_{}/02_descriptive_sentiment.xlsx".format(sent, p['currmodel']))

"""
###################### Table 3: sentiment over time, by topic and by top publishers ######################
"""

# groupby Newspaper, year, and topics, then aggregate and rename, over time
df_aggr_publisher = df_long[['year', 'Newspaper', 'DomTopic_arti_arti_id', 'mean']]\
    .groupby(['Newspaper', 'DomTopic_arti_arti_id', 'year'])\
    .agg({'Newspaper': 'count', 'mean': ['mean', 'count', 'std']}).reset_index()
# make readable column names
df_aggr_publisher.columns = df_aggr_publisher.columns.map('_'.join)

# replace everything in brackets from Newspaper
df_aggr_publisher['Newspaper_'] = df_aggr_publisher.Newspaper_.\
    replace(to_replace='\([^)]*\)', value='', regex=True).\
    str.strip()

# make a variable showing the overall count of articles by publisher and merge
df_aggr_publisher_totalcount = df_aggr_publisher.groupby(['Newspaper_']).\
                                 agg({'Newspaper_count': 'sum'}).\
                                 rename(columns={'Newspaper_count': 'Newspaper_totalcount'}).reset_index()
df_aggr_publisher = pandas.merge(df_aggr_publisher,
                                 df_aggr_publisher_totalcount,
                                 on=['Newspaper_']
                                 )

# filter by x largest Newspaper (exclude empty Newspaper name)
df_aggr_publisher_topn_help = df_aggr_publisher_totalcount[df_aggr_publisher_totalcount['Newspaper_'].str.len() > 0]\
    .nlargest(15, 'Newspaper_totalcount')
topn_publishers = list(set(df_aggr_publisher_topn_help.Newspaper_.to_list())) # unique list with topn publishers
df_aggr_publisher_topn = df_aggr_publisher[df_aggr_publisher['Newspaper_'].isin(topn_publishers)]

# prepare columns for export
df_aggr_publisher_topn = df_aggr_publisher_topn.rename(columns={'mean_mean': 'sentiment_mean',
                                                                'mean_count': 'sentiment_n',
                                                                'mean_std': 'sentiment_std',
                                                                'year_': 'year',
                                                                'DomTopic_arti_arti_id_': 'topic',
                                                                'Newspaper_': 'Newspaper'})

# export as excel
df_aggr_publisher_topn.to_excel(path_project + "tables/{}/model_{}/03_sentiment_overtime_byTopNPublish_byTopic.xlsx"\
                                .format(sent, p['currmodel']),
                                index=False)

"""
#################### Table 4: descriptives, by year, on full sample AND by topic ####################
"""

## on full sample
# group by date only, calculate mean, std and n, and aggregate by year
df_wide_fullsample_mean = df_long.groupby(['month'])[['mean']].mean()\
    .groupby(pandas.Grouper(freq='Y')).mean().reset_index().rename(columns={'month': 'year'})
df_wide_fullsample_std = df_long.groupby(['month'])[['std']].mean()\
    .groupby(pandas.Grouper(freq='Y')).mean().reset_index().rename(columns={'month': 'year'})
df_wide_fullsample_n = df_long.groupby(['month'])[['count']].sum()\
    .groupby(pandas.Grouper(freq='Y')).sum().reset_index().rename(columns={'month': 'year'})

## by each topic
# group by topics, calculate mean, std and n, and reshape long to wide, and finally aggregate by year
df_wide_bytopics_mean = df_long.groupby(['DomTopic_arti_arti_id', 'month'])[['mean']].mean().reset_index().pivot(
    index='month', columns='DomTopic_arti_arti_id', values='mean')\
    .groupby(pandas.Grouper(freq='Y')).mean().reset_index().rename(columns={'month': 'year'})
df_wide_bytopics_std = df_long.groupby(['DomTopic_arti_arti_id', 'month'])[['std']].mean().reset_index().pivot(
    index='month', columns='DomTopic_arti_arti_id', values='std')\
    .groupby(pandas.Grouper(freq='Y')).mean().reset_index().rename(columns={'month': 'year'})
df_wide_bytopics_n = df_long.groupby(['DomTopic_arti_arti_id', 'month'])[['count']].sum().reset_index().pivot(
    index='month', columns='DomTopic_arti_arti_id', values='count')\
    .groupby(pandas.Grouper(freq='Y')).sum().reset_index().rename(columns={'month': 'year'})

## Reformat dates
df_wide_fullsample_mean['year'] = pandas.DatetimeIndex(df_wide_fullsample_mean.iloc[:, 0]).year
df_wide_fullsample_std['year'] = pandas.DatetimeIndex(df_wide_fullsample_std.iloc[:, 0]).year
df_wide_fullsample_n['year'] = pandas.DatetimeIndex(df_wide_fullsample_n.iloc[:, 0]).year
df_wide_bytopics_mean['year'] = pandas.DatetimeIndex(df_wide_bytopics_mean.iloc[:, 0]).year
df_wide_bytopics_std['year'] = pandas.DatetimeIndex(df_wide_bytopics_std.iloc[:, 0]).year
df_wide_bytopics_n['year'] = pandas.DatetimeIndex(df_wide_bytopics_n.iloc[:, 0]).year

# before appending all dfs, define a help-id to sort properly later
df_wide_fullsample_mean['help_id_'] = 1
df_wide_fullsample_std['help_id_'] = 2
df_wide_fullsample_n['help_id_'] = 1
df_wide_bytopics_mean['help_id_'] = 1
df_wide_bytopics_std['help_id_'] = 2
df_wide_bytopics_n['help_id_'] = 1

# create empty df-template with years (each year 2x) and help_id_ (1, 2, 2); we merge all dfs to this
yr_list = sorted(list(range(2009, 2020))*2)
help_id_list = list(range(1, 3))*len(list(range(2009, 2020)))

list_to_df = []
for y in range(2009, 2020):
    list_to_df.append([y, 1])
    list_to_df.append([y, 2])

df_merged_stats = pandas.DataFrame(list_to_df, columns=['year', 'help_id_'])

# format numbers, to string and add brackets to std
df_wide_fullsample_mean['mean'] = df_wide_fullsample_mean['mean'].astype(float).map('{:,.2f}'.format)
df_wide_fullsample_std['std'] = df_wide_fullsample_std['std'].astype(float).map('{:,.2f}'.format)
df_wide_fullsample_std['std'] = df_wide_fullsample_std['std'].apply(lambda x: '(' + x + ')')

for col in df_wide_bytopics_mean.columns[1:-1]:
    df_wide_bytopics_mean[col] = df_wide_bytopics_mean[col].astype(float).map('{:,.2f}'.format)
    df_wide_bytopics_std[col] = df_wide_bytopics_std[col].astype(float).map('{:,.2f}'.format)
    df_wide_bytopics_std[col] = df_wide_bytopics_std[col].apply(lambda x: '(' + x + ')')

## merge all fullsample dfs (mean, std)
# merge fullsample mean
df_merged_stats = pandas.merge(df_merged_stats,
                               df_wide_fullsample_mean,
                               on=['year', 'help_id_'],
                               how='left'
                               )
# merge fullsample std, fill the NaNs in mean by std and drop std column
df_merged_stats = pandas.merge(df_merged_stats,
                               df_wide_fullsample_std,
                               on=['year', 'help_id_'],
                               how='left'
                               )
df_merged_stats['mean'] = df_merged_stats['mean'].fillna(df_merged_stats['std'])
df_merged_stats = df_merged_stats.drop('std', axis=1)

## merge all bytopic dfs (mean, std)
# merge bytopic mean
df_merged_stats = pandas.merge(df_merged_stats,
                               df_wide_bytopics_mean,
                               on=['year', 'help_id_'],
                               how='left'
                               )
# merge bytopic std, fill the NaNs in mean by std and drop std column
df_merged_stats = pandas.merge(df_merged_stats,
                               df_wide_bytopics_std,
                               on=['year', 'help_id_'],
                               how='left'
                               )

# loop over topic columns in df left-hand-side, fill na by right-hand-side topic columns, and clean up
for i in [c for c in df_merged_stats.columns if 'x' in c]:
    df_merged_stats[i] = df_merged_stats[i].fillna(df_merged_stats[i[:-1]+'y'])
    df_merged_stats.drop(i[:-1]+'y', axis=1, inplace=True)
    df_merged_stats.rename(columns={i: i[:-2]}, inplace=True)


## merge all fullsample dfs (n)
# merge fullsample n
df_merged_stats = pandas.merge(df_merged_stats,
                               df_wide_fullsample_n,
                               on=['year', 'help_id_'],
                               how='left'
                               )

## merge all bytopic dfs (n)
# merge bytopic n
df_merged_stats = pandas.merge(df_merged_stats,
                               df_wide_bytopics_n,
                               on=['year', 'help_id_'],
                               how='left'
                               )

# replace NaN with None
df_merged_stats = df_merged_stats.where(pandas.notnull(df_merged_stats), None)

# rename columns
for i in [c for c in df_merged_stats.columns if 'x' in c]:
    df_merged_stats = df_merged_stats.rename(columns={i: i[:-1]+'mean',
                                                      i[:-1]+'y': i[:-1]+'n'})

# reorder and select columns
select_cols = ['year', 'mean', 'count']
select_cols.extend(sorted(df_merged_stats.columns)[:-4])
df_merged_stats = df_merged_stats[select_cols]

# export as excel
df_merged_stats.to_excel(path_project + "tables/{}/model_{}/04_descriptives_sentiment_count_std_full_and_bytopic.xlsx".format(sent, p['currmodel']),
                         index=False)

print('done')

###
