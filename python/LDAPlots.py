import pandas, dtale
from python.ConfigUser import path_data, path_project
from python.params import params as p
import matplotlib.pyplot as plt
import numpy as np

"""
Graph 1: Sentiment score over time, by topics
Graph 2: Frequency analysis, publication trend of articles, over time (Fritsche, Mejia)
Graph 3: Frequency analysis, publication trend of topics, over time (Fritsche, Mejia)
Graph 4: Frequency analysis, publisher bias, over time (Mejia) TODO
Graph 5: Frequency analysis, Annual total with 3-years-average, by topic, over time (Melton) TODO
Graph 6: Positive and negative sentiment with net 3-years-average, by topic, over time (Melton) TODO
Graph 7: Scatterplot number articles vs. sentiment polarity, over topics (Mejia) TODO
Graph 8: Trends in sentiment polarity, frequency of 3 sentiments (pos, neg, neutr), over time (Mejia)
Graph 9: Barplot percentage shares of sentiment polarities for selected publishers (Mejia)
Graph 10: Ratio sentiment, over time, demeaned ... TODO
Graph 11: Barplot, how many articles have a sentiment score and how many not ... TODO
"""

# Todo: look up/implement LDAvis
# https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/

# unpack POStag type, lda_levels to run lda on, lda_level to get domtopic from
POStag_type, lda_level_fit, lda_level_domtopic = p['POStag'], p['lda_level_fit'], p['lda_level_domtopic']

# Load long file
print('Loading lda_results_{}_l.csv'.format(POStag_type))
df_long = pandas.read_csv(path_data + 'csv/lda_results_{}_l.csv'.format(POStag_type),
                          sep='\t', na_filter=False)

# Select articles
df_long = df_long[df_long['Art_unique'] == 1][['DomTopic_arti_arti_id', 'year', 'quarter', 'month', 'sentiscore_mean']]

# convert dtypes
df_long['month'] = pandas.to_datetime(df_long['month'], format='%Y-%m')
df_long['sentiscore_mean'] = pandas.to_numeric(df_long['sentiscore_mean'], errors='coerce')

# select date range
df_long = df_long[(df_long['month'] >= '2007-1-1')]

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
plt.show()
plt.savefig(path_project + 'graphs/01_sentiscore_bytopic_m_{}_noa{}_nob{}.png'.format(p['POStag'],
                                                                                      p['no_below'], p['no_above']))

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
plt.show()
plt.savefig(path_project + 'graphs/01_sentiscore_bytopic_q_{}_noa{}_nob{}.png'.format(p['POStag'],
                                                                                      p['no_below'],
                                                                                      p['no_above']))

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
plt.show()
plt.savefig(path_project + 'graphs/01_sentiscore_bytopic_y_{}_noa{}_nob{}.png'.format(p['POStag'],
                                                                                      p['no_below'], p['no_above']))


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
ax.bar(df_senti_freq_y.iloc[:,0], df_senti_freq_y.iloc[:, 1], width=width)
ax.set_ylabel('absolute frequency')
ax.set_title('Absolute frequency of articles over time')
ax.xaxis.set_ticks(np.arange(df_senti_freq_y.iloc[:,0].min(), df_senti_freq_y.iloc[:,0].max()+1, 1))
# Access bars
rects = ax.patches
# Define labels
labels = df_senti_freq_y.iloc[:,1]
# Loop over bars and add label
for rect, label in zip(rects, labels):
    ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 2, label, ha='center', va='bottom')

plt.savefig(path_project + 'graphs/02_absfreqArt_y_{}_noa{}_nob{}.png'.format(p['POStag'],
                                                                              p['no_below'], p['no_above']))


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

plt.show()
plt.savefig(path_project + 'graphs/03_absfreqArt_bytopic_m_{}_noa{}_nob{}.png'.format(p['POStag'],
                                                                                      p['no_below'], p['no_above']))


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
plt.show()
plt.savefig(path_project + 'graphs/03_absfreqArt_bytopic_q_{}_noa{}_nob{}.png'.format(p['POStag'],
                                                                                      p['no_below'], p['no_above']))


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
plt.show()
plt.savefig(path_project + 'graphs/03_absfreqArt_bytopic_y_{}_noa{}_nob{}.png'.format(p['POStag'],
                                                                                      p['no_below'], p['no_above']))


#####










