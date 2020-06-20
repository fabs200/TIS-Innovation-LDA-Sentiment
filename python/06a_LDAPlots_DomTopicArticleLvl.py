import pandas, os, time, dtale
from python.ConfigUser import path_data, path_project
from python._HelpFunctions import filter_sentiment_params
from python.params import params as p
import matplotlib.pyplot as plt
import numpy as np
import warnings

"""
------------------------------------------
06c_LDAPlots_DomTopicArticleLvl.py - Overview of graphs
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

# Ignore some warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# unpack POStag type, lda_levels to run lda on, lda_level to get domtopic from
POStag, lda_level_fit, sent = p['POStag'], p['lda_level_fit'], p['sentiment_list']
if 'article' in p['lda_level_domtopic']: lda_level_domtopic = 'article'

# create folder in graphs with currmodel
os.makedirs(path_project + "graph/{}/model_{}".format(sent, p['currmodel']), exist_ok=True)
# create folder in currmodel with specified lda_level_domtopic
os.makedirs(path_project + "graph/{}/model_{}/{}".format(sent, p['currmodel'], lda_level_domtopic), exist_ok=True)

# Load long file (sentence-level)
print('Loading lda_results_{}_l.csv'.format(p['currmodel']))
df_long = pandas.read_csv(path_data + 'csv/lda_results_{}_l.csv'.format(p['currmodel']), sep='\t', na_filter=False)

# Rename target sentiment variable
df_long = df_long.rename(columns={'ss_{}_mean'.format(sent): 'sentiscore_mean'})
df_long['sentiscore_mean'] = pandas.to_numeric(df_long['sentiscore_mean'],errors='coerce')

# Filter df_long
df_long = filter_sentiment_params(df_long, sent)

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
df_long = df_long[(df_long['month'] >= '2009-1-1')]
df_long = df_long[(df_long['month'] <= '2020-1-1')]

# replace everything in brackets from Newspaper
df_long['Newspaper'] = df_long.Newspaper.replace(to_replace='\([^)]*\)', value='', regex=True).str.strip()


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
df_aggr_y.plot()

# # Graph by month
# fig = plt.figure(figsize=(10,5))
# ax = fig.add_axes([0.05, 0.1, 0.79, 0.79])
# for i in range(len(df_aggr_m.columns)):
#     ax.plot(df_aggr_m.iloc[:, i], marker='.', label='topic ' + str(df_aggr_m.iloc[:, i].name))
#     ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
# ax.axhline(linewidth=1, color='grey', alpha=.5)
# plt.title('Sentiment score over time, by topics\n'
#           'POStag: {}, frequency: monthly, no_below: {}, no_above: {}'.format(p['POStag'],
#                                                                               p['no_below'], p['no_above']))
# plt.savefig(path_project + 'graph/{}/model_{}/{}/01_sentiscore_bytopics_m.png'.format(sent,
#                                                                                       p['currmodel'],
#                                                                                       lda_level_domtopic))
# plt.show(block=False)
# time.sleep(1.5)
# plt.close('all')
#
# # Graph by quarter
# fig = plt.figure(figsize=(10,5))
# ax = fig.add_axes([0.05, 0.1, 0.79, 0.79])
# for i in range(len(df_aggr_q.columns)):
#     ax.plot(df_aggr_q.iloc[:, i], marker='.', label='topic ' + str(df_aggr_q.iloc[:, i].name))
#     ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
# ax.axhline(linewidth=1, color='grey', alpha=.5)
# plt.title('Sentiment score over time, by topics\n'
#           'POStag: {}, frequency: quarterly, no_below: {}, no_above: {}'.format(p['POStag'],
#                                                                                 p['no_below'],
#                                                                                 p['no_above']))
# plt.savefig(path_project + 'graph/{}/model_{}/{}/01_sentiscore_bytopics_q.png'.format(sent,
#                                                                                       p['currmodel'],
#                                                                                       lda_level_domtopic))
# plt.show(block=False)
# time.sleep(1.5)
# plt.close('all')
#
#
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
# df_aggr_q = df_senti_freq_agg.groupby(pandas.Grouper(freq='Q')).sum().reset_index().rename(columns={'month': 'quarter'})
# df_aggr_y = df_senti_freq_agg.groupby(pandas.Grouper(freq='Y')).sum().reset_index().rename(columns={'month': 'year'})
#
# ### Reformat dates
# # month
# df_aggr_m['month'] = pandas.DatetimeIndex(df_aggr_m.iloc[:,0])
# df_aggr_m['year'] = pandas.DatetimeIndex(df_aggr_m.iloc[:,0]).year
# df_aggr_m['month'] = pandas.DatetimeIndex(df_aggr_m.iloc[:,0]).month
# df_aggr_m['month'] = df_aggr_m.year.map(str) + '-' + df_aggr_m.month.map(str)
# df_aggr_m = df_aggr_m.drop('year', axis=1)
# # quarter
# df_aggr_q['quarter'] = pandas.DatetimeIndex(df_aggr_q.iloc[:,0])
# df_aggr_q['year'] = pandas.DatetimeIndex(df_aggr_q.iloc[:,0]).year
# df_aggr_q['quarter'] = pandas.DatetimeIndex(df_aggr_q.iloc[:,0]).quarter
# df_aggr_q['quarter'] = df_aggr_q.year.map(str) + '-' + df_aggr_q.quarter.map(str)
# df_aggr_q = df_aggr_q.drop('year', axis=1)
# # year
# df_aggr_y['year'] = pandas.DatetimeIndex(df_aggr_y.iloc[:,0]).year
#
# # calculate percentages of all columns
# for i in range(1, len(df_aggr_m.columns)):
#     df_aggr_m.iloc[:, i] = df_aggr_m.iloc[:, i].div(np.array(df_aggr_m.iloc[:, i]).sum(), axis=0).apply(lambda x: 100*x)
# for i in range(1, len(df_aggr_q.columns)):
#     df_aggr_q.iloc[:, i] = df_aggr_q.iloc[:, i].div(np.array(df_aggr_q.iloc[:, i]).sum(), axis=0).apply(lambda x: 100*x)
# for i in range(1, len(df_aggr_y.columns)):
#     df_aggr_y.iloc[:, i] = df_aggr_y.iloc[:, i].div(np.array(df_aggr_y.iloc[:, i]).sum(), axis=0).apply(lambda x: 100*x)
#
# # Graph line by month
# fig = plt.figure(figsize=(10,5))
# ax = fig.add_axes([0.05, 0.1, 0.79, 0.79])
# for i in range(1, len(df_aggr_m.columns)):
#     ax.plot(df_aggr_m.iloc[:, i], marker='.', label='topic ' + str(df_aggr_m.iloc[:, i].name))
# ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
# ax.set_xticks(range(len(df_aggr_m.iloc[:, 0])))
# ax.set_xticklabels(df_aggr_m.iloc[:, 0])
# idx = 1
# for label in ax.xaxis.get_ticklabels():
#     if (idx % 6)==0 or (idx==1):
#         label.set_rotation(90)
#     else:
#         label.set_visible(False)
#     idx+=1
# ax.set_ylabel('relative frequency in percent')
# plt.title('Relative frequency of articles with sentiment score over time, by topics\n'
#           'POStag: {}, frequency: monthly, no_below: {}, no_above: {}'.format(p['POStag'],
#                                                                               p['no_below'], p['no_above']))
# plt.savefig(path_project + 'graph/{}/model_{}/{}/16_relfreqArt_bytopic_m.png'.format(sent,
#                                                                                      p['currmodel'],
#                                                                                      lda_level_domtopic))
# plt.show(block=False)
# time.sleep(1.5)
# plt.close('all')
#
# # Graph line by quarter
# fig = plt.figure(figsize=(10,5))
# ax = fig.add_axes([0.05, 0.1, 0.79, 0.79])
# for i in range(1, len(df_aggr_q.columns)):
#     ax.plot(df_aggr_q.iloc[:, i], marker='.', label='topic ' + str(df_aggr_q.iloc[:, i].name))
# ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
# ax.set_xticks(range(len(df_aggr_q.iloc[:, 0])))
# ax.set_xticklabels(df_aggr_q.iloc[:, 0])
# idx = 1
# for label in ax.xaxis.get_ticklabels():
#     if (idx % 6)==0 or (idx==1):
#         label.set_rotation(90)
#     else:
#         label.set_visible(False)
#     idx+=1
# ax.set_ylabel('relative frequency in percent')
# plt.title('Relative frequency of articles with sentiment score over time, by topics\n'
#           'POStag: {}, frequency: quarterly, no_below: {}, no_above: {}'.format(p['POStag'],
#                                                                                 p['no_below'], p['no_above']))
# plt.savefig(path_project + 'graph/{}/model_{}/{}/16_relfreqArt_bytopic_q.png'.format(sent,
#                                                                                      p['currmodel'],
#                                                                                      lda_level_domtopic))
# plt.show(block=False)
# time.sleep(1.5)
# plt.close('all')
#
#
# # Graph line by year
# fig = plt.figure(figsize=(10,5))
# ax = fig.add_axes([0.05, 0.1, 0.79, 0.79])
# for i in range(1, len(df_aggr_y.columns)):
#     ax.plot(df_aggr_y.iloc[:, i], marker='.', label='topic ' + str(df_aggr_y.iloc[:, i].name))
# ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
# ax.set_xticks(range(len(df_aggr_y.index)))
# ax.set_xticklabels([str(x) for x in df_aggr_y.iloc[:, 0].values])
# ax.set_ylabel('relative frequency in percent')
# plt.title('Relative frequency of articles with sentiment score over time, by topics\n'
#           'POStag: {}, frequency: yearly, no_below: {}, no_above: {}'.format(p['POStag'],
#                                                                              p['no_below'], p['no_above']))
# plt.savefig(path_project + 'graph/{}/model_{}/{}/16_relfreqArt_bytopic_y.png'.format(sent,
#                                                                                      p['currmodel'],
#                                                                                      lda_level_domtopic))
# plt.show(block=False)
# time.sleep(1.5)
# plt.close('all')
#
#
# #####
#
#
#
#
#
#
#

