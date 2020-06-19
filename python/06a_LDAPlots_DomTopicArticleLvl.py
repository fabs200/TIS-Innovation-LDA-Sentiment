import pandas, os, time, dtale
from python.ConfigUser import path_data, path_project
from python.params import params as p
import matplotlib.pyplot as plt
import numpy as np
import warnings

"""
------------------------------------------
06a_LDAPlots_DomTopicArticleLvl.py - Overview of graphs
------------------------------------------

### Run this script if 'article' is specified in lda_level_domtopic

Graph 1: Sentiment score over time, by topics
Graph 2: Frequency analysis, publication trend of articles, over time (Fritsche, Mejia)
Graph 3: Frequency analysis, publication trend of topics, over time (Fritsche, Mejia)
Graph 4: Frequency analysis, publisher bias, over time (Mejia)
Graph 5: Frequency analysis, Annual total with 3-years-average, by topic, over time (Melton)
Graph 6: Frequency analysis, barplot, frequency of published articles of top publishers
Graph 7: Barplot percentage shares of topics for selected publishers (stacked/not stacked) (Mejia)
Graph 8: Histogram Sentiment
Graph 9: Histogram Sentiment by year
Graph 10: Boxplot sentiments by year
Graph 11: Barplot, how many articles have a sentiment score and how many not ...
Graph 12: Barplot, how many articles have a sentiment score and how many not, by year
Graph 13: Valid and non-valid sentiments by different lengths of articles
Graph 14: Average length of articles with valid and non-valid sentiments by different lengths of articles
Graph 15: (as Graph 2, but shares), relative frequency analysis, publication trend of articles, over time
Graph 16: (as Graph 3, but shares), relative frequency analysis, publication trend of topics, over time
Graph 17: Barplot percentage shares of articles by year, stacked TODO
Graph 18: Ratio sentiment, over time, demeaned ... TODO

not:
Graph: Positive and negative sentiment with net 3-years-average, by topic, over time (Melton) TODO
Graph: Scatterplot number articles vs. sentiment polarity, over topics (Mejia) TODO
Graph: Trends in sentiment polarity, frequency of 3 sentiments (pos, neg, neutr), over time (Mejia)
"""

# select sentiment type ['sepldefault', 'seplmodified', 'sentiwsdefault', 'sentifinal']
sent = 'sentifinal'

# Ignore some warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# unpack POStag type, lda_levels to run lda on, lda_level to get domtopic from
POStag, lda_level_fit = p['POStag'], p['lda_level_fit']
if 'article' in p['lda_level_domtopic']: lda_level_domtopic = 'article'

# create folder in graphs with currmodel
os.makedirs(path_project + "graph/{}/model_{}".format(sent, p['currmodel']), exist_ok=True)
# create folder in currmodel with specified lda_level_domtopic
os.makedirs(path_project + "graph/{}/model_{}/{}".format(sent, p['currmodel'], lda_level_domtopic), exist_ok=True)

# Load long file (sentence-level)
print('Loading lda_results_{}_l.csv'.format(p['currmodel']))
df_long = pandas.read_csv(path_data + 'csv/lda_results_{}_l.csv'.format(p['currmodel']), sep='\t', na_filter=False)

#drop short articles
df_long['articles_text_lenght']= df_long['articles_text'].str.len()
df_long= df_long.drop(df_long[df_long.articles_text_lenght<300].index)

#drop short sentences
df_long['sentences_for_sentiment_lenght']= df_long['sentences_for_sentiment'].str.len()
df_long= df_long.drop(df_long[df_long.sentences_for_sentiment_lenght<50].index)

# drop articles with low probability of assigned dominant topic
drop_prob_below = .7
df_long['DomTopic_arti_arti_prob'] = pandas.to_numeric(df_long['DomTopic_arti_arti_prob'])
df_long = df_long.drop(df_long[df_long.DomTopic_arti_arti_prob < drop_prob_below].index)

# set main sentiscore_mean, rename and to numeric
df_long['sentiscore_mean'] = df_long['ss_{}_mean'.format(sent)]
df_long['sentiscore_mean'] = pandas.to_numeric(df_long['sentiscore_mean'], errors='coerce')

#drop sentences with (relatively) neutral sentiment score (either =0 or in range(-.1, .1)
drop_senti_below = .05
drop_senti_above = -.05

df_long = df_long.drop(df_long[(df_long.sentiscore_mean < drop_senti_below) & (df_long.sentiscore_mean > drop_senti_above)].index)
# keep values between range
#df_long = df_long[df_long['sentiscore_mean'].between(-.1, .1, inclusive=False)]

# calculate average sentiment per article and merge to df_long
df_long = pandas.merge(df_long.drop('sentiscore_mean', axis=1),
                       df_long.groupby('Art_ID', as_index=False).sentiscore_mean.mean(),
                       how='left', on=['Art_ID'])

# Select articles and columns
df_long = df_long[df_long['Art_unique'] == 1][['DomTopic_arti_arti_id',
                                               'year', 'quarter', 'month',
                                               'Newspaper', 'sentiscore_mean', 'articles_text']]

# convert dtypes
df_long['month'] = pandas.to_datetime(df_long['month'], format='%Y-%m')
df_long['sentiscore_mean'] = pandas.to_numeric(df_long['sentiscore_mean'], errors='coerce')
df_long['sentiscore_mean'] = pandas.to_numeric(df_long['sentiscore_mean'], errors='coerce')

# select date range
#ToDo: data range not the same on graph. on graph 2010-2020
df_long = df_long[(df_long['month'] >= '2009-1-1')]
df_long = df_long[(df_long['month'] <= '2020-1-1')]

# replace everything in brackets from Newspaper
df_long['Newspaper'] = df_long.Newspaper.replace(to_replace='\([^)]*\)', value='', regex=True).str.strip()

# df_long.to_excel(path_data + 'df_long_tocheckarticles.xlsx')



"""
###################### Graph 1: Sentiment score over time, by topics ######################
"""

# group by topics and reshape long to wide to make plottable
df_wide_bytopics = df_long.groupby(['DomTopic_arti_arti_id', 'month'])[['sentiscore_mean']].mean().reset_index().pivot(
    index='month', columns='DomTopic_arti_arti_id', values='sentiscore_mean')

# make aggregation available
df_aggr_m = df_wide_bytopics.groupby(pandas.Grouper(freq='M')).mean()
df_aggr_q = df_wide_bytopics.groupby(pandas.Grouper(freq='Q')).mean()
df_aggr_y = df_wide_bytopics.groupby(pandas.Grouper(freq='Y')).mean()

# plot
# df_aggr_m.plot()
# df_aggr_q.plot()
# df_aggr_y.plot()

# Graph by month
fig = plt.figure(figsize=(10,5))
ax = fig.add_axes([0.05, 0.1, 0.79, 0.79])
for i in range(len(df_aggr_m.columns)):
    ax.plot(df_aggr_m.iloc[:, i], marker='.', label='topic ' + str(df_aggr_m.iloc[:, i].name))
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
ax.axhline(linewidth=1, color='grey', alpha=.5)
plt.title('Sentiment score over time, by topics\n'
          'POStag: {}, frequency: monthly, no_below: {}, no_above: {}'.format(p['POStag'],
                                                                              p['no_below'], p['no_above']))
plt.savefig(path_project + 'graph/{}/model_{}/{}/01_sentiscore_bytopics_m.png'.format(sent,
                                                                                      p['currmodel'],
                                                                                      lda_level_domtopic))
plt.show(block=False)
time.sleep(1.5)
plt.close('all')

# Graph by quarter
fig = plt.figure(figsize=(10,5))
ax = fig.add_axes([0.05, 0.1, 0.79, 0.79])
for i in range(len(df_aggr_q.columns)):
    ax.plot(df_aggr_q.iloc[:, i], marker='.', label='topic ' + str(df_aggr_q.iloc[:, i].name))
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
ax.axhline(linewidth=1, color='grey', alpha=.5)
plt.title('Sentiment score over time, by topics\n'
          'POStag: {}, frequency: quarterly, no_below: {}, no_above: {}'.format(p['POStag'],
                                                                                p['no_below'],
                                                                                p['no_above']))
plt.savefig(path_project + 'graph/{}/model_{}/{}/01_sentiscore_bytopics_q.png'.format(sent,
                                                                                      p['currmodel'],
                                                                                      lda_level_domtopic))
plt.show(block=False)
time.sleep(1.5)
plt.close('all')


# Graph by year
fig = plt.figure(figsize=(10,5))
ax = fig.add_axes([0.05, 0.1, 0.79, 0.79])
for i in range(len(df_aggr_y.columns)):
    ax.plot(df_aggr_y.iloc[:, i], marker='.', label='topic ' + str(df_aggr_y.iloc[:, i].name))
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
ax.axhline(linewidth=1, color='grey', alpha=.5)
plt.title('Sentiment score over time, by topics\n'
          'POStag: {}, frequency: yearly, no_below: {}, no_above: {}'.format(p['POStag'],
                                                                             p['no_below'], p['no_above']))
# plt.savefig(path_project + 'graph/{}/model_{}/{}/01_sentiscore_bytopics_y.png'.format(sent,
#                                                                                       p['currmodel'],
#                                                                                       lda_level_domtopic))
plt.show(block=False)
time.sleep(1.5)
plt.close('all')



# import numpy
# def smoothListGaussian(list, strippedXs=False, degree=5):
#     window = degree*2-1
#     weight = numpy.array([1.0]*window)
#     weightGauss = []
#     for i in range(window):
#         i = i-degree+1
#         frac = i/float(window)
#         gauss = 1/(numpy.exp((4*(frac))**2))
#         weightGauss.append(gauss)
#     weight = numpy.array(weightGauss)*weight
#     smoothed = [0.0]*(len(list)-window)
#     for i in range(len(smoothed)):
#         smoothed[i] = sum(numpy.array(list[i:i+window])*weight)/sum(weight)
#     return smoothed


"""
###################### Graph 2: Frequency analysis, publication trend of articles over time (Fritsche, Mejia) ##########
"""

# group by year
df_senti_freq_y = df_long.groupby(['year'])[['sentiscore_mean']].count().reset_index()

# Graph bar by year
width = .75
fig = plt.figure(figsize=(10, 5))
ax = fig.add_axes([0.05, 0.1, 0.79, 0.79])
ax.bar(df_senti_freq_y.iloc[:,0], df_senti_freq_y.iloc[:, 1], color='#004488', width=width)
ax.set_ylabel('absolute frequency')
ax.set_title('Absolute frequency of articles over time\n'
             'POStag: {}, frequency: yearly, no_below: {}, no_above: {}'.format(p['POStag'],
                                                                                p['no_below'], p['no_above']))
ax.xaxis.set_ticks(np.arange(df_senti_freq_y.iloc[:,0].min(), df_senti_freq_y.iloc[:,0].max()+1, 1))
# Access bars
rects = ax.patches
# Define labels
labels = df_senti_freq_y.iloc[:,1]
# Loop over bars and add label
for rect, label in zip(rects, labels):
    ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 2, label, ha='center', va='bottom')

plt.savefig(path_project + 'graph/{}/model_{}/{}/02_absfreqArt_y.png'.format(sent,
                                                                             p['currmodel'],
                                                                             lda_level_domtopic))
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
df_aggr_m = df_senti_freq_agg.groupby(pandas.Grouper(freq='M')).sum().reset_index()
df_aggr_q = df_senti_freq_agg.groupby(pandas.Grouper(freq='Q')).sum().reset_index().rename(columns={'month': 'quarter'})
df_aggr_y = df_senti_freq_agg.groupby(pandas.Grouper(freq='Y')).sum().reset_index().rename(columns={'month': 'year'})

### Reformat dates
# month
df_aggr_m['month'] = pandas.DatetimeIndex(df_aggr_m.iloc[:,0])
df_aggr_m['year'] = pandas.DatetimeIndex(df_aggr_m.iloc[:,0]).year
df_aggr_m['month'] = pandas.DatetimeIndex(df_aggr_m.iloc[:,0]).month
df_aggr_m['month'] = df_aggr_m.year.map(str) + '-' + df_aggr_m.month.map(str)
df_aggr_m = df_aggr_m.drop('year', axis=1)
# quarter
df_aggr_q['quarter'] = pandas.DatetimeIndex(df_aggr_q.iloc[:,0])
df_aggr_q['year'] = pandas.DatetimeIndex(df_aggr_q.iloc[:,0]).year
df_aggr_q['quarter'] = pandas.DatetimeIndex(df_aggr_q.iloc[:,0]).quarter
df_aggr_q['quarter'] = df_aggr_q.year.map(str) + '-' + df_aggr_q.quarter.map(str)
df_aggr_q = df_aggr_q.drop('year', axis=1)
# year
df_aggr_y['year'] = pandas.DatetimeIndex(df_aggr_y.iloc[:,0]).year

# Graph line by month
fig = plt.figure(figsize=(10,5))
ax = fig.add_axes([0.05, 0.1, 0.79, 0.79])
for i in range(1, len(df_aggr_m.columns)):
    ax.plot(df_aggr_m.iloc[:, i], marker='.', label='topic ' + str(df_aggr_m.iloc[:, i].name))
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
ax.set_xticks(range(len(df_aggr_m.iloc[:, 0])))
ax.set_xticklabels(df_aggr_m.iloc[:, 0])
idx = 1
for label in ax.xaxis.get_ticklabels():
    if (idx % 6)==0 or (idx==1):
        label.set_rotation(90)
    else:
        label.set_visible(False)
    idx+=1
plt.title('Absolute frequency of articles with sentiment score over time, by topics\n'
          'POStag: {}, frequency: monthly, no_below: {}, no_above: {}'.format(p['POStag'],
                                                                              p['no_below'], p['no_above']))
plt.savefig(path_project + 'graph/{}/model_{}/{}/03_absfreqArt_bytopic_m.png'.format(sent,
                                                                                     p['currmodel'],
                                                                                     lda_level_domtopic))
plt.show(block=False)
time.sleep(1.5)
plt.close('all')

# Graph line by quarter
fig = plt.figure(figsize=(10,5))
ax = fig.add_axes([0.05, 0.1, 0.79, 0.79])
for i in range(1, len(df_aggr_q.columns)):
    ax.plot(df_aggr_q.iloc[:, i], marker='.', label='topic ' + str(df_aggr_q.iloc[:, i].name))
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
ax.set_xticks(range(len(df_aggr_q.iloc[:, 0])))
ax.set_xticklabels(df_aggr_q.iloc[:, 0])
idx = 1
for label in ax.xaxis.get_ticklabels():
    if (idx % 6)==0 or (idx==1):
        label.set_rotation(90)
    else:
        label.set_visible(False)
    idx+=1
plt.title('Absolute frequency of articles with sentiment score over time, by topics\n'
          'POStag: {}, frequency: quarterly, no_below: {}, no_above: {}'.format(p['POStag'],
                                                                                p['no_below'], p['no_above']))
plt.savefig(path_project + 'graph/{}/model_{}/{}/03_absfreqArt_bytopic_q.png'.format(sent,
                                                                                     p['currmodel'],
                                                                                     lda_level_domtopic))
plt.show(block=False)
time.sleep(1.5)
plt.close('all')


# Graph line by year
fig = plt.figure(figsize=(10,5))
ax = fig.add_axes([0.05, 0.1, 0.79, 0.79])
for i in range(1, len(df_aggr_y.columns)):
    ax.plot(df_aggr_y.iloc[:, i], marker='.', label='topic ' + str(df_aggr_y.iloc[:, i].name))
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
ax.set_xticks(range(len(df_aggr_y.index)))
ax.set_xticklabels([str(x) for x in df_aggr_y.iloc[:, 0].values])
plt.title('Absolute frequency of articles with sentiment score over time, by topics\n'
          'POStag: {}, frequency: yearly, no_below: {}, no_above: {}'.format(p['POStag'],
                                                                             p['no_below'], p['no_above']))
plt.savefig(path_project + 'graph/{}/model_{}/{}/03_absfreqArt_bytopic_y.png'.format(sent,
                                                                                     p['currmodel'],
                                                                                     lda_level_domtopic))
plt.show(block=False)
time.sleep(1.5)
plt.close('all')


"""
Graph 4: Frequency analysis, publisher bias, over time (Mejia)
"""

# groupby Newspaper, year, aggregate and rename, over time
df_aggr_publisher_y = df_long[['year', 'Newspaper', 'sentiscore_mean']]\
    .groupby(['year', 'Newspaper'])\
    .agg({'Newspaper': 'count', 'sentiscore_mean': ['mean', 'count', 'std']}).reset_index()
df_aggr_publisher_y.columns = df_aggr_publisher_y.columns.map('_'.join)

# groupby Newspaper, year, aggregate, drop Newspaper with less than 7 count, sort
df_aggr_publisher = df_aggr_publisher_y.groupby(['Newspaper_']).\
    agg({'Newspaper_': 'count',
         'sentiscore_mean_mean': ['mean', 'std'],
         'sentiscore_mean_std': 'mean'}).reset_index()
df_aggr_publisher.columns = df_aggr_publisher.columns.map('_'.join)
df_aggr_publisher = df_aggr_publisher[df_aggr_publisher['Newspaper__count']>5]

# replace everything in brackets from Newspaper
df_aggr_publisher['Newspaper__'] = df_aggr_publisher.Newspaper__.\
    replace(to_replace='\([^)]*\)', value='', regex=True).\
    str.strip()

# Plot Publisher's sentiment score, with Stderr
ax = df_aggr_publisher[['Newspaper__', 'sentiscore_mean_mean_mean']].\
    sort_values(by='sentiscore_mean_mean_mean').\
    plot(kind='barh', figsize=(7, 7.5), zorder=2, width=0.8, color='#004488')
ax.tick_params(axis="both", which="both", bottom="off", top="on", labelbottom="off",
               left="off", right="off", labelleft="on")
# set up ytitlelabels
ax.set_yticklabels(df_aggr_publisher.Newspaper__)
# set up xticks and xtickslabels
ax.set_xticks(np.around(np.arange(-.6, .81, step=0.2), decimals=1))
ax.set_xticklabels(np.around(np.arange(-.6, .81, step=0.2), decimals=1))
# Draw vertical axis lines
for tick in ax.get_xticks():
    ax.axvline(x=tick, linestyle='solid', alpha=0.25, color='#eeeeee', zorder=1)
# Draw horizontal axis lines
for tick in ax.get_yticks():
    ax.axhline(y=tick, linestyle='solid', alpha=0.25, color='#eeeeee', zorder=1)
# no legend
ax.get_legend().remove()
# For each bar: Place a label
for rect in ax.patches:
    # Get X and Y placement of label from rect.
    x_value = rect.get_width()
    y_value = rect.get_y() + rect.get_height() / 2

    # Number of points between bar and label. Change to your liking.
    space = 5
    # Vertical alignment for positive values
    ha = 'left'

    # If value of bar is negative: Place label left of bar
    if x_value < 0:
        # Invert space to place label to the left
        space *= -1
        # Horizontally align label at right
        ha = 'right'

    # Use X value as label and format number with one decimal place
    label = "{:.2f}".format(x_value)

    # Create annotation
    plt.annotate(
        label,                      # Use `label` as label
        (x_value, y_value),         # Place label at end of the bar
        xytext=(space, 0),          # Horizontally shift label by `space`
        textcoords="offset points", # Interpret `xytext` as offset in points
        va='center',                # Vertically center label
        ha=ha)                      # Horizontally align label differently for
                                    # positive and negative values.

ax.set_xlabel("Sentiment score")
plt.title('Average sentiment score by main publishers\n'
          'POStag: {}, frequency: yearly,\nno_below: {}, no_above: {}'.format(p['POStag'],
                                                                              p['no_below'], p['no_above']))
plt.tight_layout()
plt.savefig(path_project + 'graph/{}/model_{}/{}/04a_sentiscore_bypublisher.png'.format(sent,
                                                                                        p['currmodel'],
                                                                                        lda_level_domtopic))
plt.show(block=False)
time.sleep(1.5)
plt.close('all')



# Plot Publisher's sentiment score, with Stderr
ax = df_aggr_publisher[['sentiscore_mean_mean_mean', 'sentiscore_mean_mean_std']].\
    sort_values(by='sentiscore_mean_mean_mean').\
    plot.barh(figsize=(7, 7.75), zorder=2, width=0.65, xerr='sentiscore_mean_mean_std', color='#004488')
ax.tick_params(axis="both", which="both", bottom="off", top="on", labelbottom="off",
               left="off", right="off", labelleft="on")
# set up ytitlelabels
ax.set_yticklabels(df_aggr_publisher.Newspaper__)
# set up xticks and xtickslabels
ax.set_xticks(np.around(np.arange(-.8, .81, step=0.2), decimals=1))
ax.set_xticklabels(np.around(np.arange(-.8, .81, step=0.2), decimals=1))
# Draw vertical axis lines
vals = ax.get_xticks()
for tick in vals:
    ax.axvline(x=tick, linestyle='solid', alpha=0.25, color='#eeeeee', zorder=1)
# no legend
ax.get_legend().remove()
ax.set_xlabel("Sentiment score")
plt.title('Average sentiment score of publisher\n'
          'POStag: {}, frequency: yearly,\nno_below: {}, no_above: {}'.format(p['POStag'],
                                                                              p['no_below'], p['no_above']))
plt.tight_layout()
plt.savefig(path_project + 'graph/{}/model_{}/{}/04b_absfreqArt_bytopic_y.png'.format(sent,
                                                                                      p['currmodel'],
                                                                                      lda_level_domtopic))
plt.show(block=False)
time.sleep(1.5)
plt.close('all')


"""
############ Graph 5: Frequency analysis, Annual total with 3-years-average, by topic, over time (Melton) ############
"""

# group by topics and reshape long to wide to make plottable
df_senti_freq_agg = df_long.groupby(['DomTopic_arti_arti_id', 'month'])[['sentiscore_mean']].count().reset_index()\
    .pivot(index='month', columns='DomTopic_arti_arti_id', values='sentiscore_mean')

# make aggregation available
# df_aggr_m = df_senti_freq_agg.groupby(pandas.Grouper(freq='M')).sum().reset_index()
df_aggr_q = df_senti_freq_agg.groupby(pandas.Grouper(freq='Q')).sum().reset_index().rename(columns={'month': 'quarter'})
df_aggr_y = df_senti_freq_agg.groupby(pandas.Grouper(freq='Y')).sum().reset_index().rename(columns={'month': 'year'})

# reformat dates (no month here anymore as too granular)
# quarter
df_aggr_q['quarter'] = pandas.DatetimeIndex(df_aggr_q.iloc[:,0])
df_aggr_q['year'] = pandas.DatetimeIndex(df_aggr_q.iloc[:,0]).year
df_aggr_q['quarter'] = pandas.DatetimeIndex(df_aggr_q.iloc[:,0]).quarter
df_aggr_q['quarter'] = df_aggr_q.year.map(str) + '-' + df_aggr_q.quarter.map(str)
df_aggr_q = df_aggr_q.drop('year', axis=1)
# year
df_aggr_y['year'] = pandas.DatetimeIndex(df_aggr_y.iloc[:,0]).year

# Make column names string if pandas reads as numeric
df_aggr_q.columns = df_aggr_q.columns.astype(str)
df_aggr_y.columns = df_aggr_y.columns.astype(str)

# before calculating avgs, retrieve all topics
topics = df_aggr_q.columns[1:]

# 3-years-average, quarterly (window=3*4)
for i in range(1, len(df_aggr_q.columns)):
    topic = df_aggr_q.columns[i]
    df_aggr_q['{}_{}'.format(topic, '3y-avg')] = df_aggr_q.iloc[:, i].rolling(window=12).mean()

for t, topic in enumerate(topics):
    # Bar plot, quarterly, with 3-years-avg
    ax = df_aggr_q['{}'.format(topic)].\
        plot(kind='bar', figsize=(5, 4.5), zorder=2, width=0.8, color='#004488')
    ax.tick_params(axis="both", which="both", bottom="off", top="on", labelbottom="off",
                   left="off", right="off", labelleft="on")
    # line with 3-years-avg
    df_aggr_q['{}_3y-avg'.format(topic)].plot(secondary_y=False, color='red')
    # set up xticks and xtickslabels
    ax.set_xticks(range(len(df_aggr_q.iloc[:, 0])))
    ax.set_xticklabels(df_aggr_q.iloc[:, 0])
    idx = 1
    for label in ax.xaxis.get_ticklabels():
        if (idx % 4)==0 or (idx==1):
            label.set_rotation(90)
        else:
            label.set_visible(False)
        idx += 1
    # title
    plt.title('Frequency of articles of topic {} with 3-years-average\n'
              'POStag: {}, frequency: quarterly,\nno_below: {}, no_above: {}'.format(t, p['POStag'],
                                                                                        p['no_below'], p['no_above']))
    plt.tight_layout()
    plt.savefig(path_project + 'graph/{}/model_{}/{}/05_freq_q_topic{}_withline.png'.format(sent,
                                                                                            p['currmodel'],
                                                                                            lda_level_domtopic,
                                                                                            str(t)))
    plt.show(block=False)
    time.sleep(1.5)
    plt.close('all')



# 3-years-average, yearly
for i in range(1, len(df_aggr_y.columns)):
    topic = df_aggr_y.columns[i]
    df_aggr_y['{}_{}'.format(topic, '3y-avg')] = df_aggr_y.iloc[:, i].rolling(window=3, win_type='bohman').mean()

for t, topic in enumerate(topics):
    # Bar plot, quarterly, with 3-years-avg
    ax = df_aggr_y['{}'.format(topic)].\
        plot(kind='bar', figsize=(5, 4.5), zorder=2, width=0.8, color='#004488')
    ax.tick_params(axis="both", which="both", bottom="off", top="on", labelbottom="off",
                   left="off", right="off", labelleft="on")
    # line with 3-years-avg
    df_aggr_y['{}_3y-avg'.format(topic)].plot(secondary_y=False, color='red')
    # set up xticks and xtickslabels
    ax.set_xticks(range(len(df_aggr_y.iloc[:, 0])))
    ax.set_xticklabels(df_aggr_y.iloc[:, 0])
    idx = 1
    for label in ax.xaxis.get_ticklabels():
        if (idx % 2 != 0) or (idx==1):
            label.set_rotation(0) #90
        else:
            label.set_visible(False)
        idx += 1
    # title
    plt.title('Frequency of articles of topic {} with 3-years-average,\n'
              'POStag: {}, frequency: yearly,\nno_below: {}, no_above: {}'.format(t, p['POStag'],
                                                                                     p['no_below'], p['no_above']))
    plt.tight_layout()
    plt.savefig(path_project + 'graph/{}/model_{}/{}/05_freq_y_topic{}_withline.png'.format(sent,
                                                                                            p['currmodel'],
                                                                                            lda_level_domtopic,
                                                                                            str(t)))
    plt.show(block=False)
    time.sleep(1.5)
    plt.close('all')



"""
Graph 6: Frequency analysis, barplot, frequency of published articles of top publishers TODO
"""

# groupby Newspapyer, year, aggregate and rename, over time
df_aggr_publisher_y = df_long[['year', 'Newspaper', 'sentiscore_mean']]\
    .groupby(['year', 'Newspaper'])\
    .agg({'Newspaper': 'count', 'sentiscore_mean': ['mean', 'count', 'std']}).reset_index()
df_aggr_publisher_y.columns = df_aggr_publisher_y.columns.map('_'.join)

df_aggr_publisher_y = pandas.merge(df_aggr_publisher_y,
                                   df_aggr_publisher_y.groupby(['Newspaper_']).agg(
                                       {'Newspaper_': 'count'}).rename(columns={'Newspaper_': 'Newspaper_total'}),
                                   how='left', on=['Newspaper_'])
# Keep top publishers
# df_aggr_publisher_y['Newspaper_'].nunique()
df_aggr_publisher_y = df_aggr_publisher_y[df_aggr_publisher_y['Newspaper_total']>7]

# aggregarte by year
df_aggr_publisher_agg = df_aggr_publisher_y[['Newspaper_', 'sentiscore_mean_mean', 'Newspaper_total']].\
    groupby(['Newspaper_']).mean().reset_index()

# Plot top Publisher
ax = df_aggr_publisher_agg[['Newspaper_', 'Newspaper_total']].\
    sort_values(by='Newspaper_total').\
    plot(kind='barh', figsize=(5, 7), zorder=2, width=0.8, color='#004488')
ax.tick_params(axis="both", which="both", bottom="off", top="on", labelbottom="off",
               left="off", right="off", labelleft="on")
# set up ytitlelabels
ax.set_yticklabels(df_aggr_publisher_agg.Newspaper_)
# set up xticks and xtickslabels
ax.set_xticks(np.around(np.arange(0, df_aggr_publisher_agg.Newspaper_total.max()+3, step=2), decimals=1))
ax.set_xticklabels(np.around(np.arange(0, df_aggr_publisher_agg.Newspaper_total.max()+3, step=2), decimals=1))
# Draw vertical axis lines
for tick in ax.get_xticks():
    ax.axvline(x=tick, linestyle='solid', alpha=0.25, color='#eeeeee', zorder=1)
# Draw horizontal axis lines
for tick in ax.get_yticks():
    ax.axhline(y=tick, linestyle='solid', alpha=0.25, color='#eeeeee', zorder=1)
# no legend
ax.get_legend().remove()
# For each bar: Place a label
for rect in ax.patches:
    # Get X and Y placement of label from rect.
    x_value = rect.get_width()
    y_value = rect.get_y() + rect.get_height() / 2

    # Number of points between bar and label. Change to your liking.
    space = 5
    # Vertical alignment for positive values
    ha = 'left'

    # If value of bar is negative: Place label left of bar
    if x_value < 0:
        # Invert space to place label to the left
        space *= -1
        # Horizontally align label at right
        ha = 'right'

    # Use X value as label and format number with one decimal place
    label = "{:.0f}".format(x_value)

    # Create annotation
    plt.annotate(
        label,                      # Use `label` as label
        (x_value, y_value),         # Place label at end of the bar
        xytext=(space, 0),          # Horizontally shift label by `space`
        textcoords="offset points", # Interpret `xytext` as offset in points
        va='center',                # Vertically center label
        ha=ha)                      # Horizontally align label differently for
                                    # positive and negative values.

ax.set_xlabel("total number of articles")
plt.title('Total number of articles by \nmain publishers\n'
          'POStag: {}, frequency: yearly,\nno_below: {}, no_above: {}'.format(p['POStag'],
                                                                              p['no_below'], p['no_above']))
plt.tight_layout()
plt.savefig(path_project + 'graph/{}/model_{}/{}/06_totalarticles_bypublisher.png'.format(sent,
                                                                                          p['currmodel'],
                                                                                          lda_level_domtopic))
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

# plot stacked bar plot
barWidth = 0.85
topics = df_publishers_bytopics_t['topics'].to_list()
publishers = df_publishers_bytopics_t.columns[1:]
# plot stacked bars
fig = plt.figure(figsize=(10, 4.5))
ax = fig.add_axes([0.08, 0.2, 0.8, 0.65])
for i in range(0, len(topics)):
    if i==0:
        ax.bar(publishers, df_publishers_bytopics_t.iloc[i].to_list()[1:])
    else:
        ax.bar(publishers, df_publishers_bytopics_t.iloc[i].to_list()[1:],
                bottom=df_publishers_bytopics_t.iloc[0:i].sum().to_list()[1:])
# legend
ax.legend(topics, title='topics', bbox_to_anchor=(1.1, .75), ncol=1, borderaxespad=0.,
          fontsize='small', loc='upper right', )
# Custom x axis
plt.xticks(rotation=90, size=7)
plt.xlabel('Newspaper')
# Custom y axis
plt.ylabel("percentage shares in topics")
plt.tight_layout()
plt.title('Topics by publishers\n'
          'POStag: {}, frequency: monthly, no_below: {}, no_above: {}'.format(p['POStag'],
                                                                              p['no_below'], p['no_above']))
plt.savefig(path_project + 'graph/{}/model_{}/{}/07_topics_by_publishers_stacked.png'.format(sent,
                                                                                             p['currmodel'],
                                                                                             lda_level_domtopic),
            bbox_inches='tight')
plt.show(block=False)
time.sleep(1.5)
plt.close('all')


"""
Graph 8: Histogram, Sentiment
"""

# histogram
plt.hist(df_long['sentiscore_mean'].to_list(), bins=20, color='#004488', edgecolor='b', alpha=0.65)
# mean
plt.axvline(df_long['sentiscore_mean'].mean(),
            color='r', linestyle='--', linewidth=.5)
plt.text(df_long['sentiscore_mean'].mean()*1.1, 10,
         'Mean: {:.2f}'.format(df_long['sentiscore_mean'].mean()), color='r')
# median
plt.axvline(df_long['sentiscore_mean'].mean(),
            color='lime', linestyle=':', linewidth=.5)
plt.text(df_long['sentiscore_mean'].mean()*1.1, 20,
         'Median: {:.2f}'.format(df_long['sentiscore_mean'].mean()), color='lime')
plt.title('Histogram sentiment score\n'
          'POStag: {}, no_below: {}, no_above: {}'.format(p['POStag'], p['no_below'], p['no_above']))
plt.savefig(path_project + 'graph/{}/model_{}/{}/08_hist_sentiment.png'.format(sent,
                                                                               p['currmodel'],
                                                                               lda_level_domtopic),
            bbox_inches='tight')
plt.show(block=False)
time.sleep(1.5)
plt.close('all')


"""
Graph 9: Histogram Sentiment by year
"""

# extract years
years = np.sort(df_long['year'].unique())
# set up graph and title
fig, axs = plt.subplots(4, 4, constrained_layout=True)
fig.suptitle('Sentiment score by year\n'
          'POStag: {}, no_below: {}, no_above: {}'.format(p['POStag'], p['no_below'], p['no_above']), fontsize=12)
# loop over axis and list and plot hist
for i, ax in enumerate(axs.reshape(-1)):
    if i < len(years):
        ax.hist(df_long[df_long['year'] == years[i]]['sentiscore_mean'], bins=20)
        ax.set_title(years[i], fontsize=12)
plt.savefig(path_project + 'graph/{}/model_{}/{}/09_hist_sentiment_byyears.png'.format(sent,
                                                                                       p['currmodel'],
                                                                                       lda_level_domtopic),
            bbox_inches='tight')
plt.show(block=False)
time.sleep(1.5)
plt.close('all')

"""
Graph 10: Boxplot sentiments over year
"""
boxplot = df_long[['year', 'sentiscore_mean']].boxplot(column=['sentiscore_mean'], by='year')
plt.savefig(path_project + 'graph/{}/model_{}/{}/10_boxplot_sentiment_overyears.png'.format(sent,
                                                                                            p['currmodel'],
                                                                                            lda_level_domtopic),
            bbox_inches='tight')
plt.show(block=False)
time.sleep(1.5)
plt.close('all')

"""
Graph 11: Barplot, how many articles have a sentiment score and how many not
"""

# generate dummy indicating filled sentiscore =1 or not =2 (nan)
df_long['D_filledsent'] = df_long['sentiscore_mean'].notnull().astype('int').replace(0, 2)
# group by dummy
df_filledsent = df_long.groupby(['D_filledsent']).count().reset_index()
# label
label = ['with sentiment', 'without sentiment']

# bar plot
fig, ax = plt.subplots(figsize=(5, 4.5))
# Define bar width. We'll use this to offset the second bar.
x, bar_width = 0, 0.5
# 1. bars and number on bars
b1 = ax.bar(x, df_filledsent.loc[df_filledsent['D_filledsent']==1, 'Newspaper'], width=bar_width)
ax.text(x, df_filledsent.loc[df_filledsent['D_filledsent']==1, 'Newspaper'] + 10,
        str(df_filledsent.loc[df_filledsent['D_filledsent']==1, 'Newspaper'].values[0]),
        fontweight='bold')
# 2. bars and number on bars
b2 = ax.bar(x + bar_width, df_filledsent.loc[df_filledsent['D_filledsent']==2, 'Newspaper'], width=bar_width)
ax.text(x+bar_width, df_filledsent.loc[df_filledsent['D_filledsent']==2, 'Newspaper'] + 10,
        str(df_filledsent.loc[df_filledsent['D_filledsent']==2, 'Newspaper'].values[0]),
        fontweight='bold')
# label
plt.xticks([x, x+bar_width], label)
plt.ylabel('frequency articles')
plt.title('Bar plot of articles w/o valid sentiments\n'
          'POStag: {}, no_below: {}, no_above: {}'.format(p['POStag'], p['no_below'], p['no_above']))
plt.tight_layout()
plt.savefig(path_project + 'graph/{}/model_{}/{}/11_barplot_sentavailability.png'.format(sent,
                                                                                         p['currmodel'],
                                                                                         lda_level_domtopic),
            bbox_inches='tight')
plt.show(block=False)
time.sleep(1.5)
plt.close('all')

"""
Graph 12: Barplot, how many articles have a sentiment score and how many not, by year
"""

# generate dummy indicating filled sentiscore =1 or not =2 (nan)
df_long['D_filledsent'] = df_long['sentiscore_mean'].notnull().astype('int').replace(0, 2)
# group by dummy
df_filledsent_yr = df_long.groupby(['D_filledsent', 'year']).count().unstack(fill_value=0).stack().reset_index()
df_filledsent_yr = df_filledsent_yr[['D_filledsent', 'year', 'Newspaper']]
# df_filledsent_yr['year'] = df_filledsent_yr['year'].astype('str')
df_filledsent_yr['D_filledsent'] = df_filledsent_yr['D_filledsent'].astype('str').replace('1', 'filled').replace('2', 'empty')

# label
label = ['with sentiment', 'without sentiment']
# years
years = df_filledsent_yr.loc[df_filledsent_yr['D_filledsent']=='filled', 'year'].to_numpy()

# bar plot
fig, ax = plt.subplots(figsize=(8, 4.5))
# Define bar width. We'll use this to offset the second bar.
bar_width = .4
# 1. bars and number on bars, filled
b1 = ax.bar(years-bar_width/2,
            df_filledsent_yr.loc[df_filledsent_yr['D_filledsent']=='filled', 'Newspaper'].to_list(),
            width=bar_width)
# loop over each pair of x- and y-value and annotate b1 bars
for i in range(len(years)):
    # Create annotation
    x_val, y_val = years[i], df_filledsent_yr.loc[df_filledsent_yr['D_filledsent']=='filled', 'Newspaper'].to_list()[i]
    plt.annotate("{:.0f}".format(y_val), (x_val-1.25*bar_width, y_val+5),
                 xytext=(space, 0), textcoords="offset points", va='center', ha=ha, size=7)
# 2. bars and number on bars, empty
b2 = ax.bar(years+bar_width/2,
            df_filledsent_yr.loc[df_filledsent_yr['D_filledsent']=='empty', 'Newspaper'].to_list(),
            width=bar_width)
# loop over each pair of x- and y-value and annotate b2 bars
for i in range(len(years)):
    # Create annotation
    x_val, y_val = years[i], df_filledsent_yr.loc[df_filledsent_yr['D_filledsent']=='empty', 'Newspaper'].to_list()[i]
    plt.annotate("{:.0f}".format(y_val), (x_val, y_val+5),
                 xytext=(space, 0), textcoords="offset points", va='center', ha=ha, size=7)
# label
ax.set_xticks(years)
# legend, axis labels
plt.legend([b1, b2], ['valid sentiment', 'missing sentiment'], loc='upper left')
plt.ylabel('frequency articles')
# For each bar: Place a label

plt.title('Bar plot of articles w/o valid sentiments by year\n'
          'POStag: {}, no_below: {}, no_above: {}'.format(p['POStag'], p['no_below'], p['no_above']))
plt.tight_layout()
plt.savefig(path_project + 'graph/{}/model_{}/{}/12_barplot_sentavailability_byyear.png'.format(sent,
                                                                                                p['currmodel'],
                                                                                                lda_level_domtopic),
            bbox_inches='tight')
plt.show(block=False)
time.sleep(1.5)
plt.close('all')


"""
Graph 13: Valid and non-valid sentiments by different lengths of articles
"""

# generate dummy indicating filled sentiscore =1 or not =2 (nan)
df_long['D_filledsent'] = df_long['sentiscore_mean'].notnull().astype('int').replace(0, 2)
# get deciles
deciles = 9
df_long['articles_len'] = df_long.articles_text.apply(lambda x: len(x))
df_long['articles_len_dc'] = pandas.qcut(df_long['articles_len'], deciles, labels=False)
df_filledsent_dc = df_long[['articles_len_dc', 'articles_len', 'D_filledsent']].\
    groupby(['articles_len_dc', 'D_filledsent']).count().unstack(fill_value=0).stack().reset_index()

# label
label = ['with sentiment', 'without sentiment']

# bar plot
fig, ax = plt.subplots(figsize=(8, 4.5))
# Define bar width. We'll use this to offset the second bar.
bar_width = .4
# 1. bars and number on bars, filled
b1 = ax.bar(df_filledsent_dc.loc[df_filledsent_dc['D_filledsent']==1]['articles_len_dc'].to_numpy()-bar_width/2,
            df_filledsent_dc.loc[df_filledsent_dc['D_filledsent']==1]['articles_len'].to_list(),
            width=bar_width)
# loop over each pair of x- and y-value and annotate b1 bars
for i in range(deciles):
    # Create annotation
    x_val, y_val = i-bar_width, df_filledsent_dc.loc[df_filledsent_dc['D_filledsent']==1, 'articles_len'].to_list()[i]
    plt.annotate("{:.0f}".format(y_val), (x_val, y_val+5),
                 xytext=(space, 0), textcoords="offset points", va='center', ha=ha, size=7)
# 2. bars and number on bars, empty
b2 = ax.bar(df_filledsent_dc.loc[df_filledsent_dc['D_filledsent']==2]['articles_len_dc'].to_numpy()+bar_width/2,
            df_filledsent_dc.loc[df_filledsent_dc['D_filledsent']==2]['articles_len'].to_list(),
            width=bar_width)
# loop over each pair of x- and y-value and annotate b2 bars
for i in range(deciles):
    # Create annotation
    x_val, y_val = i, df_filledsent_dc.loc[df_filledsent_dc['D_filledsent']==2, 'articles_len'].to_list()[i]
    plt.annotate("{:.0f}".format(y_val), (x_val, y_val+5),
                 xytext=(space, 0), textcoords="offset points", va='center', ha=ha, size=7)
# label
ax.set_xticks(df_filledsent_dc.loc[df_filledsent_dc['D_filledsent']==1]['articles_len_dc'].to_numpy())
ax.set_xticklabels(labels=10*(1+df_filledsent_dc.loc[df_filledsent_dc['D_filledsent']==1]['articles_len_dc'].to_numpy()))

# legend, axis labels
plt.legend([b1, b2], ['valid sentiment', 'missing sentiment'], loc='lower left')
plt.ylabel('frequency articles')
plt.xlabel('deciles of length of articles')
# For each bar: Place a label

plt.title('Bar plot of deciles of article lengths w/o valid sentiments by year\n'
          'POStag: {}, no_below: {}, no_above: {}'.format(p['POStag'], p['no_below'], p['no_above']))
plt.tight_layout()
plt.savefig(path_project + 'graph/{}/model_{}/{}/13_barplot_sentavailability_deciles.png'.format(sent,
                                                                                                 p['currmodel'],
                                                                                                 lda_level_domtopic),
            bbox_inches='tight')
plt.show(block=False)
time.sleep(1.5)
plt.close('all')



"""
Graph 14: Average length of articles with valid and non-valid sentiments by different lengths of articles
"""

# generate dummy indicating filled sentiscore =1 or not =2 (nan)
df_long['D_filledsent'] = df_long['sentiscore_mean'].notnull().astype('int').replace(0, 2)
# get deciles
deciles = 9
df_long['articles_len'] = df_long.articles_text.apply(lambda x: len(x))
df_long['articles_len_dc'] = pandas.qcut(df_long['articles_len'], deciles, labels=False)
# aggregate deciles
df_filledsent_dc = df_long[['articles_len_dc', 'articles_len', 'D_filledsent']].\
    groupby(['articles_len_dc', 'D_filledsent']).count().unstack(fill_value=0).stack().reset_index().\
    rename(columns={'articles_len': 'count'})
# aggregate lengths
df_filledsent_dc_len = df_long[['articles_len_dc', 'articles_len', 'D_filledsent']].\
    groupby(['articles_len_dc', 'D_filledsent']).mean().unstack(fill_value=0).stack().reset_index().\
    rename(columns={'articles_len': 'avg_len'})
# add deciles to lengths df
df_filledsent_dc_len.insert(2, 'count', df_filledsent_dc['count'].to_list(), True)

# label
label = ['with sentiment', 'without sentiment']

# bar plot
fig, ax = plt.subplots(figsize=(8, 4.5))
# Define bar width. We'll use this to offset the second bar.
bar_width = .4
# 1. bars and number on bars, filled
b1 = ax.bar(df_filledsent_dc_len.loc[df_filledsent_dc_len['D_filledsent']==1]['articles_len_dc'].to_numpy()-bar_width/2,
            df_filledsent_dc_len.loc[df_filledsent_dc_len['D_filledsent']==1]['count'].to_list(),
            width=bar_width)
# loop over each pair of x- and y-value and annotate b1 bars
for i in range(deciles):
    # Create annotation
    x_val, y_val = i-bar_width, df_filledsent_dc_len.loc[df_filledsent_dc_len['D_filledsent']==1, 'count'].to_list()[i]
    display_val = df_filledsent_dc_len.loc[df_filledsent_dc_len['D_filledsent']==1, 'avg_len'].to_list()[i]
    plt.annotate("({:.0f})".format(display_val), (x_val, y_val+5),
                 xytext=(space, 0), textcoords="offset points", va='center', ha=ha, size=7)
# 2. bars and number on bars, empty
b2 = ax.bar(df_filledsent_dc_len.loc[df_filledsent_dc_len['D_filledsent']==2]['articles_len_dc'].to_numpy()+bar_width/2,
            df_filledsent_dc_len.loc[df_filledsent_dc_len['D_filledsent']==2]['count'].to_list(),
            width=bar_width)
# loop over each pair of x- and y-value and annotate b2 bars
for i in range(deciles):
    # Create annotation
    x_val, y_val = i, df_filledsent_dc_len.loc[df_filledsent_dc_len['D_filledsent']==2, 'count'].to_list()[i]
    display_val = df_filledsent_dc_len.loc[df_filledsent_dc_len['D_filledsent']==2, 'avg_len'].to_list()[i]
    plt.annotate("({:.0f})".format(display_val), (x_val, y_val+5),
                 xytext=(space, 0), textcoords="offset points", va='center', ha=ha, size=7)
# label
ax.set_xticks(df_filledsent_dc_len.loc[df_filledsent_dc_len['D_filledsent']==1]['articles_len_dc'].to_numpy())
ax.set_xticklabels(labels=10*(1+df_filledsent_dc_len.loc[df_filledsent_dc_len['D_filledsent']==1]['articles_len_dc'].to_numpy()))

# legend, axis labels
plt.legend([b1, b2], ['valid sentiment', 'missing sentiment'], loc='lower left')
plt.ylabel('frequency articles')
plt.xlabel('deciles of length of articles')
# For each bar: Place a label

plt.title('Bar plot of deciles of article lengths w/o valid sentiments by year,\n'
          'average length of article in brackets on bars,\n'
          'POStag: {}, no_below: {}, no_above: {}'.format(p['POStag'], p['no_below'], p['no_above']))
plt.tight_layout()
plt.savefig(path_project + 'graph/{}/model_{}/{}/14_barplot_sentavailability_deciles_withartilen.png'.format(sent,
                                                                                                             p['currmodel'],
                                                                                                             lda_level_domtopic),
            bbox_inches='tight')
plt.show(block=False)
time.sleep(1.5)
plt.close('all')

"""
Graph 15: (as Graph 2, but shares), relative frequency analysis, publication trend of articles, over time
"""
# group by year
df_senti_freq_y = df_long.groupby(['year'])[['sentiscore_mean']].count().reset_index()

# Graph bar by year
width, totalrows = .75, np.array(df_senti_freq_y.sentiscore_mean.to_list()).sum()
fig = plt.figure(figsize=(10, 5))
ax = fig.add_axes([0.05, 0.1, 0.79, 0.79])
ax.bar(df_senti_freq_y.iloc[:,0], 100*(df_senti_freq_y.iloc[:, 1]/totalrows), color='#004488', width=width)
ax.set_ylabel('relative frequency in percent')
ax.set_title('Relative frequency of articles over time\n'
             'POStag: {}, frequency: yearly, no_below: {}, no_above: {}'.format(p['POStag'],
                                                                                p['no_below'], p['no_above']))
ax.xaxis.set_ticks(np.arange(df_senti_freq_y.iloc[:,0].min(), df_senti_freq_y.iloc[:,0].max()+1, 1))
# Access bars
rects = ax.patches
# Define labels
labels = round(100*(df_senti_freq_y.iloc[:,1]/totalrows), 2)
# Loop over bars and add label
for rect, label in zip(rects, labels):
    ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(), label, ha='center', va='bottom')

plt.savefig(path_project + 'graph/{}/model_{}/{}/15_relfreqArt_y.png'.format(sent, p['currmodel'], lda_level_domtopic))
plt.show(block=False)
time.sleep(1.5)
plt.close('all')


"""
Graph 16: (as Graph 3, but shares), relative frequency analysis, publication trend of topics, over time
"""

# group by topics and reshape long to wide to make plottable
df_senti_freq_agg = df_long.groupby(['DomTopic_arti_arti_id', 'month'])[['sentiscore_mean']].count().reset_index()\
    .pivot(index='month', columns='DomTopic_arti_arti_id', values='sentiscore_mean')

# make aggregation available
df_aggr_m = df_senti_freq_agg.groupby(pandas.Grouper(freq='M')).sum().reset_index()
df_aggr_q = df_senti_freq_agg.groupby(pandas.Grouper(freq='Q')).sum().reset_index().rename(columns={'month': 'quarter'})
df_aggr_y = df_senti_freq_agg.groupby(pandas.Grouper(freq='Y')).sum().reset_index().rename(columns={'month': 'year'})

### Reformat dates
# month
df_aggr_m['month'] = pandas.DatetimeIndex(df_aggr_m.iloc[:,0])
df_aggr_m['year'] = pandas.DatetimeIndex(df_aggr_m.iloc[:,0]).year
df_aggr_m['month'] = pandas.DatetimeIndex(df_aggr_m.iloc[:,0]).month
df_aggr_m['month'] = df_aggr_m.year.map(str) + '-' + df_aggr_m.month.map(str)
df_aggr_m = df_aggr_m.drop('year', axis=1)
# quarter
df_aggr_q['quarter'] = pandas.DatetimeIndex(df_aggr_q.iloc[:,0])
df_aggr_q['year'] = pandas.DatetimeIndex(df_aggr_q.iloc[:,0]).year
df_aggr_q['quarter'] = pandas.DatetimeIndex(df_aggr_q.iloc[:,0]).quarter
df_aggr_q['quarter'] = df_aggr_q.year.map(str) + '-' + df_aggr_q.quarter.map(str)
df_aggr_q = df_aggr_q.drop('year', axis=1)
# year
df_aggr_y['year'] = pandas.DatetimeIndex(df_aggr_y.iloc[:,0]).year

# calculate percentages of all columns
for i in range(1, len(df_aggr_m.columns)):
    df_aggr_m.iloc[:, i] = df_aggr_m.iloc[:, i].div(np.array(df_aggr_m.iloc[:, i]).sum(), axis=0).apply(lambda x: 100*x)
for i in range(1, len(df_aggr_q.columns)):
    df_aggr_q.iloc[:, i] = df_aggr_q.iloc[:, i].div(np.array(df_aggr_q.iloc[:, i]).sum(), axis=0).apply(lambda x: 100*x)
for i in range(1, len(df_aggr_y.columns)):
    df_aggr_y.iloc[:, i] = df_aggr_y.iloc[:, i].div(np.array(df_aggr_y.iloc[:, i]).sum(), axis=0).apply(lambda x: 100*x)

# Graph line by month
fig = plt.figure(figsize=(10,5))
ax = fig.add_axes([0.05, 0.1, 0.79, 0.79])
for i in range(1, len(df_aggr_m.columns)):
    ax.plot(df_aggr_m.iloc[:, i], marker='.', label='topic ' + str(df_aggr_m.iloc[:, i].name))
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
ax.set_xticks(range(len(df_aggr_m.iloc[:, 0])))
ax.set_xticklabels(df_aggr_m.iloc[:, 0])
idx = 1
for label in ax.xaxis.get_ticklabels():
    if (idx % 6)==0 or (idx==1):
        label.set_rotation(90)
    else:
        label.set_visible(False)
    idx+=1
ax.set_ylabel('relative frequency in percent')
plt.title('Relative frequency of articles with sentiment score over time, by topics\n'
          'POStag: {}, frequency: monthly, no_below: {}, no_above: {}'.format(p['POStag'],
                                                                              p['no_below'], p['no_above']))
plt.savefig(path_project + 'graph/{}/model_{}/{}/16_relfreqArt_bytopic_m.png'.format(sent,
                                                                                     p['currmodel'],
                                                                                     lda_level_domtopic))
plt.show(block=False)
time.sleep(1.5)
plt.close('all')

# Graph line by quarter
fig = plt.figure(figsize=(10,5))
ax = fig.add_axes([0.05, 0.1, 0.79, 0.79])
for i in range(1, len(df_aggr_q.columns)):
    ax.plot(df_aggr_q.iloc[:, i], marker='.', label='topic ' + str(df_aggr_q.iloc[:, i].name))
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
ax.set_xticks(range(len(df_aggr_q.iloc[:, 0])))
ax.set_xticklabels(df_aggr_q.iloc[:, 0])
idx = 1
for label in ax.xaxis.get_ticklabels():
    if (idx % 6)==0 or (idx==1):
        label.set_rotation(90)
    else:
        label.set_visible(False)
    idx+=1
ax.set_ylabel('relative frequency in percent')
plt.title('Relative frequency of articles with sentiment score over time, by topics\n'
          'POStag: {}, frequency: quarterly, no_below: {}, no_above: {}'.format(p['POStag'],
                                                                                p['no_below'], p['no_above']))
plt.savefig(path_project + 'graph/{}/model_{}/{}/16_relfreqArt_bytopic_q.png'.format(sent,
                                                                                     p['currmodel'],
                                                                                     lda_level_domtopic))
plt.show(block=False)
time.sleep(1.5)
plt.close('all')


# Graph line by year
fig = plt.figure(figsize=(10,5))
ax = fig.add_axes([0.05, 0.1, 0.79, 0.79])
for i in range(1, len(df_aggr_y.columns)):
    ax.plot(df_aggr_y.iloc[:, i], marker='.', label='topic ' + str(df_aggr_y.iloc[:, i].name))
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
ax.set_xticks(range(len(df_aggr_y.index)))
ax.set_xticklabels([str(x) for x in df_aggr_y.iloc[:, 0].values])
ax.set_ylabel('relative frequency in percent')
plt.title('Relative frequency of articles with sentiment score over time, by topics\n'
          'POStag: {}, frequency: yearly, no_below: {}, no_above: {}'.format(p['POStag'],
                                                                             p['no_below'], p['no_above']))
plt.savefig(path_project + 'graph/{}/model_{}/{}/16_relfreqArt_bytopic_y.png'.format(sent,
                                                                                     p['currmodel'],
                                                                                     lda_level_domtopic))
plt.show(block=False)
time.sleep(1.5)
plt.close('all')


#####








