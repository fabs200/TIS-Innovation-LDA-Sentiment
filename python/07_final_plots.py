import pandas, os, time, dtale
from python.ConfigUser import path_data, path_project
from python._HelpFunctions import filter_sentiment_params
from python.params import params as p
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np
import warnings

"""
------------------------------------------
07_final_plots.py - Overview of graphs
------------------------------------------

### Run this script if 'article' is specified in lda_level_domtopic

Graph 1: Sentiment score over time, by topics
Graph 2: Frequency analysis, publication trend of articles over time (Fritsche, Mejia)
Graph 3: Frequency analysis, publication trend of topics, over time (Fritsche, Mejia)
Graph 7: Barplot percentage shares of topics for selected publishers, (stacked/not stacked) (Mejia)
* NEW (October 2020):
Graph 1.1: Sentiment score over time, by topics, with events
Graph 3.1: Frequency analysis, publication trend of topics over time (Fritsche, Mejia), with events
Graph 4: Sentiment by Publisher with SD
"""

# Ignore some warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Plot title
_PLOTTITLE = False

# Specify font
_FONT = 'Arial'
csfont = {'fontname': _FONT, 'size': 16}
hfont = {'fontname': _FONT}
csfont_axis = {'fontname': _FONT, 'size': 11}
legendfont = font_manager.FontProperties(family=_FONT)

# Specify topics
tc1 = 'Industry'
tc2 = 'R&D'
tc3 = 'Infrastructure'
tc4 = 'Usability'
tc5 = 'Policy'
topics = ['topics', tc1, tc2, tc3, tc4, tc5]

# Specify color palette
# _COLORS = ['colors', '#004488', 'steelblue', 'lightblue', 'lightgray', 'grey', 'k'] # 'lightgray'
_COLORS = ['colors', '#225378', '#1483D6', '#8E9AA1', '#FF7F15', '#F43E00', '#B3CDF4'] # 'lightgray'

# unpack POStag type, lda_levels to run lda on, lda_level to get domtopic from
POStag, lda_level_fit, sent = p['POStag'], p['lda_level_fit'], p['sentiment_list']
if 'article' in p['lda_level_domtopic']: lda_level_domtopic = 'article'

# create folder in graphs with currmodel
os.makedirs(path_project + "graph/{}/model_{}".format(sent, p['currmodel']), exist_ok=True)
# create folder in currmodel with specified lda_level_domtopic
os.makedirs(path_project + "graph/{}/model_{}/{}/Final/".format(sent, p['currmodel'], lda_level_domtopic), exist_ok=True)

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
                                               'Newspaper', 'sentiscore_mean', 'articles_text']]

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

"""
###################### Graph 1: Sentiment score over time, by topics ######################
"""

# group by topics and reshape long to wide to make plottable
df_wide_bytopics = df_long.groupby(['DomTopic_arti_arti_id', 'month'])[['sentiscore_mean']].mean().reset_index().pivot(
    index='month', columns='DomTopic_arti_arti_id', values='sentiscore_mean')

# make aggregation available
df_aggr_y = df_wide_bytopics.groupby(pandas.Grouper(freq='Y')).mean().reset_index().rename(columns={'month': 'year'})

### Reformat dates
df_aggr_y['year'] = pandas.DatetimeIndex(df_aggr_y.iloc[:,0]).year

# Graph line by year
# fig = plt.figure(figsize=(10,5))
# ax = fig.add_axes([0.05, 0.1, 0.79, 0.79])
fig = plt.figure(figsize=(8, 5))
ax = fig.add_axes([0.1, 0.05, .71, .85]) # [left, bottom, width, height]
for i in range(1, len(df_aggr_y.columns)):
    ax.plot(df_aggr_y.iloc[:, i], marker='.', label=topics[i], color=_COLORS[i])
ax.set_ylabel('sentiment score', **csfont_axis)
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., prop=legendfont)
ax.set_xticks(range(len(df_aggr_y.index)))
for tick in ax.get_xticklabels():
    tick.set_fontname(_FONT)
for tick in ax.get_yticklabels():
    tick.set_fontname(_FONT)
ax.set_xticklabels([str(x) for x in df_aggr_y.iloc[:, 0].values], **csfont_axis)
plt.grid(b=True, which='major', color='#F0F0F0', linestyle='-')
ax.axhline(y=0, color='#DEDEDE')
if _PLOTTITLE: plt.title('Sentiment score over time, by topics', **csfont)
for fmt in ['png', 'pdf', 'svg']:
    plt.savefig(path_project + 'graph/{}/model_{}/{}/01_sentiscore_bytopics_y_FINAL.{}'.format(sent,
                                                                                               p['currmodel'],
                                                                                               lda_level_domtopic,
                                                                                               fmt))
plt.show(block=False)
time.sleep(1.5)
plt.close('all')


"""
###################### Graph 2: Frequency analysis, publication trend of articles over time (Fritsche, Mejia) ##########
"""

# group by year
df_senti_freq_y = df_long.groupby(['year'])[['sentiscore_mean']].count().reset_index()

# Graph bar by year
width = .75
# fig = plt.figure(figsize=(10, 5))
# ax = fig.add_axes([0.1, 0.05, 0.85, 0.85])
fig = plt.figure(figsize=(7, 5))
ax = fig.add_axes([0.1, 0.05, .85, .85]) # [left, bottom, width, height]
ax.bar(df_senti_freq_y.iloc[:,0], df_senti_freq_y.iloc[:, 1], color=_COLORS[1], width=width)
ax.set_ylabel('frequency', **csfont_axis)
if _PLOTTITLE: ax.set_title('Absolute frequency of articles over time', **csfont)
ax.xaxis.set_ticks(np.arange(df_senti_freq_y.iloc[:,0].min(), df_senti_freq_y.iloc[:,0].max()+1, 1))
# Access bars
rects = ax.patches
# Define labels
labels = df_senti_freq_y.iloc[:,1]
# Loop over bars and add label
for rect, label in zip(rects, labels):
    ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 2, label, ha='center', va='bottom', **csfont_axis)
# Set axis font
for tick in ax.get_xticklabels():
    tick.set_fontname(_FONT)
for tick in ax.get_yticklabels():
    tick.set_fontname(_FONT)

for fmt in ['png', 'pdf', 'svg']:
    plt.savefig(path_project + 'graph/{}/model_{}/{}/02_absfreqArt_y_FINAL.{}'.format(sent,
                                                                                      p['currmodel'],
                                                                                      lda_level_domtopic,
                                                                                      fmt))

plt.show(block=False)
time.sleep(1.5)
plt.close('all')



"""
###################### Graph 3: Frequency analysis, publication trend of topics over time (Fritsche, Mejia) ##########
"""

# group by topics and reshape long to wide to make plottable
df_senti_freq_agg = df_long.groupby(['DomTopic_arti_arti_id', 'month'])[['sentiscore_mean']].count().reset_index()\
    .pivot(index='month', columns='DomTopic_arti_arti_id', values='sentiscore_mean')

# make aggregation available
df_aggr_y = df_senti_freq_agg.groupby(pandas.Grouper(freq='Y')).sum().reset_index().rename(columns={'month': 'year'})

### Reformat dates
df_aggr_y['year'] = pandas.DatetimeIndex(df_aggr_y.iloc[:,0]).year

# Graph line by year
fig = plt.figure(figsize=(8, 5))
ax = fig.add_axes([0.1, 0.05, .69, .85]) # [left, bottom, width, height]
for i in range(1, len(df_aggr_y.columns)):
    ax.plot(df_aggr_y.iloc[:, i], marker='.', label=topics[i], color=_COLORS[i])
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0., prop=legendfont)
ax.set_xticks(range(len(df_aggr_y.index)))
ax.set_xticklabels([str(x) for x in df_aggr_y.iloc[:, 0].values])
ax.set_ylabel('frequency', **csfont_axis)
if _PLOTTITLE: plt.title('Absolute frequency of articles with sentiment score over time, by topics', **csfont)
# Set axis font
for tick in ax.get_xticklabels():
    tick.set_fontname(_FONT)
for tick in ax.get_yticklabels():
    tick.set_fontname(_FONT)
ax.set_xticklabels([str(x) for x in df_aggr_y.iloc[:, 0].values], **csfont_axis)
plt.grid(b=True, which='major', color='#F0F0F0', linestyle='-')
ax.axhline(y=0, color='#DEDEDE')

for fmt in ['png', 'pdf', 'svg']:
    plt.savefig(path_project + 'graph/{}/model_{}/{}/03_absfreqArt_bytopic_y_FINAL.{}'.format(sent,
                                                                                              p['currmodel'],
                                                                                              lda_level_domtopic,
                                                                                              fmt))
plt.show(block=False)
time.sleep(1.5)
plt.close('all')



"""
###################### Graph 1a + 1b: Sentiment score over time, by topics ######################
Graph 1a: usability (4) und R&D (2)
Graph 1b: Policy (5), infrastructure (3) und industry (1)
"""

# group by topics and reshape long to wide to make plottable
df_wide_bytopics = df_long.groupby(['DomTopic_arti_arti_id', 'month'])[['sentiscore_mean']].mean().reset_index().pivot(
    index='month', columns='DomTopic_arti_arti_id', values='sentiscore_mean')

# make aggregation available
df_aggr_y = df_wide_bytopics.groupby(pandas.Grouper(freq='Y')).mean().reset_index().rename(columns={'month': 'year'})

### Reformat dates
df_aggr_y['year'] = pandas.DatetimeIndex(df_aggr_y.iloc[:,0]).year

### Graph 1a line by year
# fig = plt.figure(figsize=(10,5))
# ax = fig.add_axes([0.05, 0.1, 0.79, 0.79])
fig = plt.figure(figsize=(8, 5))
ax = fig.add_axes([0.1, 0.05, .71, .85]) # [left, bottom, width, height]
for i in [2, 4]:
    ax.plot(df_aggr_y.iloc[:, i], marker='.', label=topics[i], color=_COLORS[i])
ax.set_ylabel('sentiment score', **csfont_axis)
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., prop=legendfont)
ax.set_xticks(range(len(df_aggr_y.index)))
for tick in ax.get_xticklabels():
    tick.set_fontname(_FONT)
for tick in ax.get_yticklabels():
    tick.set_fontname(_FONT)
ax.set_xticklabels([str(x) for x in df_aggr_y.iloc[:, 0].values], **csfont_axis)
plt.grid(b=True, which='major', color='#F0F0F0', linestyle='-')
plt.ylim(-0.04, 0.1)

ax.axhline(y=0, color='#DEDEDE')
if _PLOTTITLE: plt.title('Sentiment score over time, by topics', **csfont)
for fmt in ['png', 'pdf', 'svg']:
    plt.savefig(path_project + 'graph/{}/model_{}/{}/01a_sentiscore_bytopics_y_FINAL.{}'.format(sent,
                                                                                               p['currmodel'],
                                                                                               lda_level_domtopic,
                                                                                               fmt))
plt.show(block=False)
time.sleep(1.5)
plt.close('all')

### Graph 1b line by year
# fig = plt.figure(figsize=(10,5))
# ax = fig.add_axes([0.05, 0.1, 0.79, 0.79])
fig = plt.figure(figsize=(8, 5))
ax = fig.add_axes([0.1, 0.05, .71, .85]) # [left, bottom, width, height]
for i in [1, 3, 5]:
    ax.plot(df_aggr_y.iloc[:, i], marker='.', label=topics[i], color=_COLORS[i])
ax.set_ylabel('sentiment score', **csfont_axis)
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., prop=legendfont)
ax.set_xticks(range(len(df_aggr_y.index)))
for tick in ax.get_xticklabels():
    tick.set_fontname(_FONT)
for tick in ax.get_yticklabels():
    tick.set_fontname(_FONT)
ax.set_xticklabels([str(x) for x in df_aggr_y.iloc[:, 0].values], **csfont_axis)
plt.grid(b=True, which='major', color='#F0F0F0', linestyle='-')
ax.axhline(y=0, color='#DEDEDE')
# plt.title('Sentiment score over time, by topics', **csfont)
for fmt in ['png', 'pdf', 'svg']:
    plt.savefig(path_project + 'graph/{}/model_{}/{}/01b_sentiscore_bytopics_y_FINAL.{}'.format(sent,
                                                                                               p['currmodel'],
                                                                                               lda_level_domtopic,
                                                                                               fmt))
plt.show(block=False)
time.sleep(1.5)
plt.close('all')


"""
###################### Graph 3a + 3b: Frequency analysis, publication trend of topics over time (Fritsche, Mejia) ##########
Graph 3a: usability (4) und R&D (2)
Graph 3b: Policy (5), infrastructure (3) und industry (1)
"""

# group by topics and reshape long to wide to make plottable
df_senti_freq_agg = df_long.groupby(['DomTopic_arti_arti_id', 'month'])[['sentiscore_mean']].count().reset_index()\
    .pivot(index='month', columns='DomTopic_arti_arti_id', values='sentiscore_mean')

# make aggregation available
df_aggr_y = df_senti_freq_agg.groupby(pandas.Grouper(freq='Y')).sum().reset_index().rename(columns={'month': 'year'})

### Reformat dates
df_aggr_y['year'] = pandas.DatetimeIndex(df_aggr_y.iloc[:,0]).year

### Graph 3a line by year
fig = plt.figure(figsize=(8, 5))
ax = fig.add_axes([0.1, 0.05, .69, .85]) # [left, bottom, width, height]
for i in [2, 4]:
    ax.plot(df_aggr_y.iloc[:, i], marker='.', label=topics[i], color=_COLORS[i])
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0., prop=legendfont)
ax.set_xticks(range(len(df_aggr_y.index)))
ax.set_xticklabels([str(x) for x in df_aggr_y.iloc[:, 0].values])
ax.set_ylabel('frequency', **csfont_axis)
if _PLOTTITLE: plt.title('Absolute frequency of articles with sentiment score over time, by topics', **csfont)
# Set axis font
for tick in ax.get_xticklabels():
    tick.set_fontname(_FONT)
for tick in ax.get_yticklabels():
    tick.set_fontname(_FONT)
ax.set_xticklabels([str(x) for x in df_aggr_y.iloc[:, 0].values], **csfont_axis)
plt.grid(b=True, which='major', color='#F0F0F0', linestyle='-')
ax.axhline(y=0, color='#DEDEDE')
for fmt in ['png', 'pdf', 'svg']:
    plt.savefig(path_project + 'graph/{}/model_{}/{}/03a_absfreqArt_bytopic_y_FINAL.{}'.format(sent,
                                                                                              p['currmodel'],
                                                                                              lda_level_domtopic,
                                                                                              fmt))
plt.show(block=False)
time.sleep(1.5)
plt.close('all')


### Graph 3a line by year
fig = plt.figure(figsize=(8, 5))
ax = fig.add_axes([0.1, 0.05, .69, .85]) # [left, bottom, width, height]
for i in [1, 3, 5]:
    ax.plot(df_aggr_y.iloc[:, i], marker='.', label=topics[i], color=_COLORS[i])
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0., prop=legendfont)
ax.set_xticks(range(len(df_aggr_y.index)))
ax.set_xticklabels([str(x) for x in df_aggr_y.iloc[:, 0].values])
ax.set_ylabel('frequency', **csfont_axis)
if _PLOTTITLE: plt.title('Absolute frequency of articles with sentiment score over time, by topics', **csfont)
# Set axis font
for tick in ax.get_xticklabels():
    tick.set_fontname(_FONT)
for tick in ax.get_yticklabels():
    tick.set_fontname(_FONT)
ax.set_xticklabels([str(x) for x in df_aggr_y.iloc[:, 0].values], **csfont_axis)
plt.grid(b=True, which='major', color='#F0F0F0', linestyle='-')
ax.axhline(y=0, color='#DEDEDE')
for fmt in ['png', 'pdf', 'svg']:
    plt.savefig(path_project + 'graph/{}/model_{}/{}/03b_absfreqArt_bytopic_y_FINAL.{}'.format(sent,
                                                                                              p['currmodel'],
                                                                                              lda_level_domtopic,
                                                                                              fmt))
plt.show(block=False)
time.sleep(1.5)
plt.close('all')

"""
Graph 7: Barplot percentage shares of topics for selected publishers, (stacked/not stacked) (Mejia)
"""

# group by topics and reshape long to wide to make plottable
df_wide_publishers_bytopics = df_long.groupby(['DomTopic_arti_arti_id', 'Newspaper']).count().\
    reset_index()[['DomTopic_arti_arti_id', 'Newspaper', 'sentiscore_mean']].\
    rename(columns={'sentiscore_mean': 'count'}).\
    pivot(index='Newspaper', columns='DomTopic_arti_arti_id', values='count').\
    fillna(0)

# calculate percentages per topic
df_wide_publishers_bytopics['sum'] = df_wide_publishers_bytopics.sum(axis=1)
df_wide_publishers_bytopics = df_wide_publishers_bytopics[df_wide_publishers_bytopics['sum']>7]
totalcols = len(df_wide_publishers_bytopics.columns)
for col in range(0, totalcols-1):
    df_wide_publishers_bytopics.iloc[:, col] = 100*(df_wide_publishers_bytopics.iloc[:, col] / df_wide_publishers_bytopics.iloc[:, totalcols-1])
df_wide_publishers_bytopics = df_wide_publishers_bytopics.drop(['sum'], axis=1)
df_wide_publishers_bytopics = df_wide_publishers_bytopics.reset_index()

# cut str at lengt
df_wide_publishers_bytopics['Newspaper'] = df_wide_publishers_bytopics['Newspaper'].str.slice(0, 14)

# transpose, rename to make plotable
df_publishers_bytopics_t = df_wide_publishers_bytopics.transpose().reset_index()
df_publishers_bytopics_t.columns = df_publishers_bytopics_t.iloc[0]
df_publishers_bytopics_t = df_publishers_bytopics_t\
    .drop(df_publishers_bytopics_t.columns[1], axis=1)\
    .drop(0)\
    .rename(columns={'Newspaper': 'topics'})

# drop selected publishers
df_publishers_bytopics_t = df_publishers_bytopics_t.drop(['BILD Bund'], axis=1)

# rename topics according to our final identified topics
for i, tp in enumerate(topics[1:]):
    df_publishers_bytopics_t.iloc[i, 0] = tp

# plot stacked bar plot
barWidth = 0.85
topics = df_publishers_bytopics_t['topics'].to_list()
publishers = df_publishers_bytopics_t.columns[1:]
# plot stacked bars
fig = plt.figure(figsize=(10, 5))
ax = fig.add_axes([0.08, 0.18, 0.75, 0.75]) # [left, bottom, width, height]
for i in range(0, len(topics)):
    if i==0:
        ax.bar(publishers, df_publishers_bytopics_t.iloc[i].to_list()[1:])
    else:
        ax.bar(publishers, df_publishers_bytopics_t.iloc[i].to_list()[1:],
                bottom=df_publishers_bytopics_t.iloc[0:i].sum().to_list()[1:])
# legend
ax.legend(topics, title='topics', bbox_to_anchor=(1.2, .85), ncol=1, borderaxespad=0.,
          fontsize='small', loc='upper right')
# Custom x axis
plt.xticks(rotation=90, size=7)
plt.xlabel('Newspaper')
# Custom y axis
plt.ylabel("percentage shares in topics")
plt.tight_layout()
if _PLOTTITLE: plt.title('Topics by selected publishers')

for fmt in ['png', 'pdf', 'svg']:
    plt.savefig(path_project + 'graph/{}/model_{}/{}/07_topics_by_publishers_stacked_FINAL.{}'.format(sent,
                                                                                                      p['currmodel'],
                                                                                                      lda_level_domtopic,
                                                                                                      fmt),
                bbox_inches='tight')
plt.show(block=False)
time.sleep(1.5)
plt.close('all')


"""
###################### Graph 1.1: Sentiment score over time, by topics, with events ######################
"""

# group by topics and reshape long to wide to make plottable
df_wide_bytopics = df_long.groupby(['DomTopic_arti_arti_id', 'month'])[['sentiscore_mean']].mean().reset_index().pivot(
    index='month', columns='DomTopic_arti_arti_id', values='sentiscore_mean')

# make aggregation available
df_aggr_y = df_wide_bytopics.groupby(pandas.Grouper(freq='Y')).mean().reset_index().rename(columns={'month': 'year'})

### Reformat dates
df_aggr_y['year'] = pandas.DatetimeIndex(df_aggr_y.iloc[:, 0]).year

### Import events
df_events = pandas.read_excel(path_project + 'events/events_final.xlsx')

# merge events
df_aggr_y = pandas.merge(df_aggr_y, df_events, on=['year'], how='outer')

# get overall maximum of sentiment -> will define location of event labels
max_sentiment = []
for i, tp in enumerate(df_aggr_y.columns):
    templist_ = df_aggr_y[[tp]].values
    max_sentiment.append(templist_.max())
max_sentiment = np.array(max_sentiment[1:-1]).max()

# Graph line by year
fig = plt.figure(figsize=(10, 5))
ax = fig.add_axes([0.1, 0.05, .71, .9]) # [left, bottom, width, height]
ax.axhline(y=0, color='#DEDEDE')
for i in range(1, len(df_aggr_y.columns)-1):
    ax.plot(df_aggr_y.iloc[:, i], marker='.', label=topics[i-1], color=_COLORS[i])
for i in range(0, len(df_aggr_y.index)):
    plt.axvline(x=i, color='gray', alpha=.3, linestyle='dashed')
    plt.text(i, max_sentiment, s=df_aggr_y.iloc[i, 6].replace('; ', '\n'), label=df_aggr_y.iloc[i, 6],
             rotation=90, fontsize=7, wrap=True,
             alpha=.7, ha='right', va='bottom')
ax.set_ylabel('sentiment score', **csfont_axis)
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., prop=legendfont)
ax.set_xticks(range(len(df_aggr_y.index)))
for tick in ax.get_xticklabels():
    tick.set_fontname(_FONT)
for tick in ax.get_yticklabels():
    tick.set_fontname(_FONT)
ax.set_xticklabels([str(x) for x in df_aggr_y.iloc[:, 0].values], **csfont_axis)
# plt.grid(b=True, which='major', color='#F0F0F0', linestyle='-')
if _PLOTTITLE: plt.title('Sentiment score over time, by topics', **csfont)

for fmt in ['png', 'pdf', 'svg']:
    plt.savefig(path_project + 'graph/{}/model_{}/{}/01_sentiscore_bytopics_y_FINAL_events.{}'.format(sent,
                                                                                                      p['currmodel'],
                                                                                                      lda_level_domtopic,
                                                                                                      fmt))
plt.show(block=False)
time.sleep(1.5)
plt.close('all')

"""
########## Graph 3.1: Frequency analysis, publication trend of topics over time (Fritsche, Mejia), with events #########
"""

# group by topics and reshape long to wide to make plottable
df_senti_freq_agg = df_long.groupby(['DomTopic_arti_arti_id', 'month'])[['sentiscore_mean']].count().reset_index()\
    .pivot(index='month', columns='DomTopic_arti_arti_id', values='sentiscore_mean')

# make aggregation available
df_aggr_y = df_senti_freq_agg.groupby(pandas.Grouper(freq='Y')).sum().reset_index().rename(columns={'month': 'year'})

### Reformat dates
df_aggr_y['year'] = pandas.DatetimeIndex(df_aggr_y.iloc[:,0]).year

### Import events
df_events = pandas.read_excel(path_project + 'events/events_final.xlsx')

# merge events
df_aggr_y = pandas.merge(df_aggr_y, df_events, on=['year'], how='outer')

# get overall maximum of sentiment -> will define location of event labels
max_frequency = []
for i, tp in enumerate(df_aggr_y.columns):
    templist_ = df_aggr_y[[tp]].values
    max_frequency.append(templist_.max())
max_frequency = np.array(max_frequency[1:-1]).max()

# Graph line by year
fig = plt.figure(figsize=(10, 5))
ax = fig.add_axes([0.1, 0.05, .71, .9]) # [left, bottom, width, height]
ax.axhline(y=0, color='#DEDEDE')
for i in range(1, len(df_aggr_y.columns)-1):
    ax.plot(df_aggr_y.iloc[:, i], marker='.', label=topics[i-1], color=_COLORS[i])
for i in range(0, len(df_aggr_y.index)):
    plt.axvline(x=i, color='gray', alpha=.3, linestyle='dashed')
    plt.text(i, max_frequency, s=df_aggr_y.iloc[i, 6].replace('; ', '\n'), label=df_aggr_y.iloc[i, 6],
             rotation=90, fontsize=7, wrap=True,
             alpha=.7, ha='right', va='bottom')
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0., prop=legendfont)
ax.set_xticks(range(len(df_aggr_y.index)))
ax.set_xticklabels([str(x) for x in df_aggr_y.iloc[:, 0].values])
ax.set_ylabel('frequency', **csfont_axis)
if _PLOTTITLE: plt.title('Absolute frequency of articles with sentiment score over time, by topics', **csfont)
# Set axis font
for tick in ax.get_xticklabels():
    tick.set_fontname(_FONT)
for tick in ax.get_yticklabels():
    tick.set_fontname(_FONT)
# ax.set_xticklabels([str(x) for x in df_aggr_y.iloc[:, 0].values], **csfont_axis)
# plt.grid(b=True, which='major', color='#F0F0F0', linestyle='-')

for fmt in ['png', 'pdf', 'svg']:
    plt.savefig(path_project + 'graph/{}/model_{}/{}/03_absfreqArt_bytopic_y_FINAL_events.{}'.format(sent,
                                                                                              p['currmodel'],
                                                                                              lda_level_domtopic,
                                                                                              fmt))
plt.show(block=False)
time.sleep(1.5)
plt.close('all')



"""
Graph 4: Sentiment by Publisher with SD
"""

# groupby Newspaper, year, aggregate and rename, over time
df_aggr_publisher = df_long[['year', 'Newspaper', 'sentiscore_mean']]\
    .groupby(['Newspaper'])\
    .agg({'Newspaper': 'count', 'sentiscore_mean': ['mean', 'count', 'std']}).reset_index()\
    .rename(columns = {'sentiscore_mean': 'sentiscore'})
# make readable column names
df_aggr_publisher.columns = df_aggr_publisher.columns.map('_'.join)

# replace everything in brackets from Newspaper
df_aggr_publisher['Newspaper_'] = df_aggr_publisher.Newspaper_.\
    replace(to_replace='\([^)]*\)', value='', regex=True).\
    str.strip()

# filter by x largest Newspaper (exclude empty Newspaper name)
df_aggr_publisher_topn = df_aggr_publisher[df_aggr_publisher['Newspaper_'].str.len() > 0].nlargest(15, 'Newspaper_count')
topn_publishers = df_aggr_publisher_topn.Newspaper_.to_list()

# Plot Publisher's sentiment score, with Stderr
ax = df_aggr_publisher_topn[['sentiscore_mean', 'sentiscore_std']].\
    sort_values(by='sentiscore_mean').\
    plot.barh(figsize=(7, 7.75), zorder=2, width=0.65, xerr='sentiscore_std', color='#004488', alpha=0.3)
ax.tick_params(axis="both", which="both", bottom="off", top="on", labelbottom="off",
               left="off", right="off", labelleft="on")
# set up ytitlelabels
ax.set_yticklabels(df_aggr_publisher_topn.Newspaper_)
# set up xticks and xtickslabels
ax.set_xticks(np.around(np.arange(-.8, .81, step=0.2), decimals=1))
ax.set_xticklabels(np.around(np.arange(-.8, .81, step=0.2), decimals=1))
# Draw vertical axis lines
vals = ax.get_xticks()
# fix "-0.0"
vals[4] = 0
for tick in vals:
    ax.axvline(x=tick, linestyle='solid', alpha=0.25, color='#eeeeee', zorder=1)
# draw black line through sentimentscore=0
ax.axvline(x=0, linestyle='solid', alpha=.5, color='black', zorder=1)
# no legend
ax.get_legend().remove()
ax.set_xlabel("Sentiment score")
# plt.title('Average sentiment score of publisher')
plt.tight_layout()
for fmt in ['png', 'pdf', 'svg']:
    plt.savefig(path_project + 'graph/{}/model_{}/{}/04_sentiscore_bypublisher_with_ci.{}'.format(sent,
                                                                                                  p['currmodel'],
                                                                                                  lda_level_domtopic,
                                                                                                  fmt))
plt.show(block=False)
time.sleep(1.5)
plt.close('all')



"""
Graph 7: Barplot percentage shares of topics for selected publishers, (stacked/not stacked) (Mejia)
"""

# group by topics and reshape long to wide to make plottable
df_wide_publishers_bytopics = df_long.groupby(['DomTopic_arti_arti_id', 'Newspaper']).count().\
    reset_index()[['DomTopic_arti_arti_id', 'Newspaper', 'sentiscore_mean']].\
    rename(columns={'sentiscore_mean': 'count'}).\
    pivot(index='Newspaper', columns='DomTopic_arti_arti_id', values='count').\
    fillna(0)

# keep publishers which are in previous Graph 4 in topn
df_wide_publishers_bytopics = df_wide_publishers_bytopics.loc[df_wide_publishers_bytopics.index.isin(topn_publishers)]

# calculate percentages per topic
df_wide_publishers_bytopics['sum'] = df_wide_publishers_bytopics.sum(axis=1)
# df_wide_publishers_bytopics = df_wide_publishers_bytopics[df_wide_publishers_bytopics['sum']>7]
totalcols = len(df_wide_publishers_bytopics.columns)
for col in range(0, totalcols-1):
    df_wide_publishers_bytopics.iloc[:, col] = 100*(df_wide_publishers_bytopics.iloc[:, col] / df_wide_publishers_bytopics.iloc[:, totalcols-1])
df_wide_publishers_bytopics = df_wide_publishers_bytopics.drop(['sum'], axis=1)
df_wide_publishers_bytopics = df_wide_publishers_bytopics.reset_index()

# cut str at lengt
df_wide_publishers_bytopics['Newspaper'] = df_wide_publishers_bytopics['Newspaper'].str.slice(0, 16)

# transpose, rename to make plotable
df_publishers_bytopics_t = df_wide_publishers_bytopics.transpose().reset_index()
df_publishers_bytopics_t.columns = df_publishers_bytopics_t.iloc[0]
df_publishers_bytopics_t = df_publishers_bytopics_t\
    .drop(df_publishers_bytopics_t.columns[1], axis=1) \
    .rename(columns={'Newspaper': 'topics'})\
    .drop(0)

# plot stacked bar plot
barWidth = 0.85
publishers = df_publishers_bytopics_t.columns[1:]
# plot stacked bars
fig = plt.figure(figsize=(10, 4.5))
ax = fig.add_axes([0.1, 0.32, 0.76, 0.62])  # [left, bottom, width, height]
for i in range(0, len(topics)):
    if i==0:
        ax.bar(publishers, df_publishers_bytopics_t.iloc[i].to_list()[1:])
    else:
        ax.bar(publishers, df_publishers_bytopics_t.iloc[i].to_list()[1:],
                bottom=df_publishers_bytopics_t.iloc[0:i].sum().to_list()[1:], color=_COLORS[i+1])
# legend
ax.legend(topics, bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0., prop=legendfont)
# Custom x axis
plt.xticks(rotation=90, size=7)
plt.xlabel('Newspaper', **csfont_axis)
# Custom y axis
plt.ylabel('percentage shares in topics', **csfont_axis)
# Set axis font
for tick in ax.get_xticklabels():
    tick.set_fontname(_FONT)
for tick in ax.get_yticklabels():
    tick.set_fontname(_FONT)
ax.set_xticklabels([str(x) for x in df_publishers_bytopics_t.columns[1:]], **csfont_axis)
plt.tight_layout()
if _PLOTTITLE: plt.title('Topics by publishers', **csfont)
# Set axis font
for tick in ax.get_xticklabels():
    tick.set_fontname(_FONT)
for tick in ax.get_yticklabels():
    tick.set_fontname(_FONT)

for fmt in ['png', 'pdf', 'svg']:
    plt.savefig(path_project + 'graph/{}/model_{}/{}/07_topics_by_publishers_stacked_FINAL.{}'.format(sent,
                                                                                                      p['currmodel'],
                                                                                                      lda_level_domtopic,
                                                                                                      fmt),
                bbox_inches='tight')
plt.show(block=False)
time.sleep(1.5)
plt.close('all')



###
