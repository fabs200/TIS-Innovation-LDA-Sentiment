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
df_long = df_long[df_long['Art_unique'] == 1][['DomTopic_arti_arti_id',
                                               'year', 'sentiscore_mean']]
# to numeric
df_long['sentiscore_mean'] = pandas.to_numeric(df_long['sentiscore_mean'], errors='coerce')

df = df_long.groupby(['DomTopic_arti_arti_id', 'year'])['sentiscore_mean'].median().reset_index()

# to integer
df['DomTopic_arti_arti_id'] = pandas.to_numeric(df['DomTopic_arti_arti_id'], errors='coerce')
df['DomTopic_arti_arti_id'] = df['DomTopic_arti_arti_id'].astype(int)

line_chart0 = plt.plot(df[df['DomTopic_arti_arti_id']==0]['year'].to_list(), df[df['DomTopic_arti_arti_id']==0]['sentiscore_mean'].to_list(),'--')
line_chart1 = plt.plot(df[df['DomTopic_arti_arti_id']==1]['year'].to_list(), df[df['DomTopic_arti_arti_id']==1]['sentiscore_mean'].to_list(),'-')
line_chart2 = plt.plot(df[df['DomTopic_arti_arti_id']==2]['year'].to_list(), df[df['DomTopic_arti_arti_id']==2]['sentiscore_mean'].to_list(),'-.')
line_chart3 = plt.plot(df[df['DomTopic_arti_arti_id']==3]['year'].to_list(), df[df['DomTopic_arti_arti_id']==3]['sentiscore_mean'].to_list(),'-')
# line_chart4 = plt.plot(df[df['DomTopic_arti_arti_id']==4]['year'].to_list(), df[df['DomTopic_arti_arti_id']==4]['sentiscore_mean'].to_list(),'--')
# line_chart5 = plt.plot(df[df['DomTopic_arti_arti_id']==5]['year'].to_list(), df[df['DomTopic_arti_arti_id']==5]['sentiscore_mean'].to_list(),'--')
# line_chart6 = plt.plot(df[df['DomTopic_arti_arti_id']==6]['year'].to_list(), df[df['DomTopic_arti_arti_id']==6]['sentiscore_mean'].to_list(),'--')
# line_chart7 = plt.plot(df[df['DomTopic_arti_arti_id']==7]['year'].to_list(), df[df['DomTopic_arti_arti_id']==7]['sentiscore_mean'].to_list(),'--')
plt.title('Topics with sentiment over time')
plt.xlabel('year')
plt.ylabel('sentiment score')
plt.legend(['topic0', 'topic1', 'topic2', 'topic3', 'topic4', 'topic5', 'toplic6', 'toplic7'], loc=4)
plt.show()

