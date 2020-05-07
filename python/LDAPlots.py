import pandas
from python.ConfigUser import path_processedarticles
from python.params import params as p
import matplotlib.pyplot as plt

# Todo: look up/implement LDAvis
# https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/

# unpack POStag type, lda_levels to run lda on, lda_level to get domtopic from
POStag_type, lda_level_fit, lda_level_domtopic = p['POStag_type'], p['lda_level_fit'], p['lda_level_domtopic']

# Load long file
print('Loading lda_results_{}_l.csv'.format(POStag_type))
df_long = pandas.read_csv(path_processedarticles + 'csv/lda_results_{}_l.csv'.format(POStag_type),
                          sep='\t', na_filter=False)

# Select articles
df_long = df_long[df_long['Art_unique'] == 1][['DomTopic_arti_arti_id', 'month', 'sentiscore_mean']]

# convert dtypes
df_long['month'] = pandas.to_datetime(df_long['month'], format='%Y-%m')
df_long['sentiscore_mean'] = pandas.to_numeric(df_long['sentiscore_mean'], errors='coerce')

# select date range
df_long = df_long[(df_long['month'] > '2007-1-1')]

# group by topics and reshape long to wide to make plottable
df_long = df_long.groupby(['DomTopic_arti_arti_id', 'month'])[['sentiscore_mean']].mean().reset_index().pivot(
    index='month', columns='DomTopic_arti_arti_id', values='sentiscore_mean')

# make aggregation available
df_aggr_m = df_long.groupby(pandas.Grouper(freq='M')).mean()
df_aggr_q = df_long.groupby(pandas.Grouper(freq='Q')).mean()
df_aggr_y = df_long.groupby(pandas.Grouper(freq='Y')).mean()

# plot
df_aggr_m.plot()
df_aggr_q.plot()
df_aggr_y.plot()

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
