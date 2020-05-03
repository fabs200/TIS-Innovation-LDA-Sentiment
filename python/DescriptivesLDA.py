import pandas
from python.ConfigUser import path_processedarticles
from python.params import params as p

# Todo: look up/implement LDAvis
# https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/

# unpack POStag type, lda_levels to run lda on, lda_level to get domtopic from
POStag_type, lda_level_fit, lda_level_domtopic = p['POStag_type'], p['lda_level_fit'], p['lda_level_domtopic']

# Load long file
print('Loading complete_for_lda_{}_l.csv'.format(POStag_type))
df_long = pandas.read_csv(path_processedarticles + 'csv/lda_results_{}_l.csv'.format(POStag_type),
                          sep='\t', na_filter=False)


