"""
Load SepL, SentiWS, process and modify and generate final Sentiment list "Sentimentdict_final.csv"
"""

import pandas, dtale
import numpy as np
from python.ConfigUser import path_data

# Load all three sentiment dictionaries
df_sepl_default = pandas.read_csv(path_data + 'Sentiment/SePL/SePL_v1.1.csv', sep=';')
df_sepl_modified = pandas.read_csv(path_data + 'Sentiment/SePL/SePL_v1.3_negated_modified.csv', sep=';')
df_sentiws_default = pandas.read_csv(path_data + 'Sentiment/SentiWS/SentiWS_final.csv', sep=';')

# check
print(df_sepl_default.describe())
print(df_sepl_modified.describe())
print(df_sentiws_default.describe())

# Select columns for sepl dfs and rename columns in df_sentiws_default according to df_sepl
df_sepl_default = df_sepl_default[['phrase', 'opinion_value']]
df_sepl_modified = df_sepl_modified[['phrase', 'opinion_value']]
df_sentiws_default = df_sentiws_default.rename(columns={'word': 'phrase'})

# 1. Identify modified sepl phrs in df_sepl_modified and keep only modified phrases
df_sepl_onlymodified = pandas.merge(df_sepl_default, df_sepl_modified,
                                    how='outer', indicator=True)
# check numbers
print('Merge df_sepl_default with df_sepl_modified\n', df_sepl_onlymodified.groupby(['_merge']).count())
# keep only modified phrs
df_sepl_onlymodified = df_sepl_onlymodified[df_sepl_onlymodified['_merge']=='right_only'][['phrase', 'opinion_value']]

# 2. Replace phrs in sepl_default by df_sentiws_default
df_sepl_withsentiws = pandas.merge(df_sepl_default, df_sentiws_default,
                                   on='phrase',
                                   how='outer', indicator=True)

# Check numbers
print('Merge df_sepl_default with df_sentiws_default\n', df_sepl_withsentiws.groupby(['_merge']).count())

# keep only ...
# a. phrs which are only in sepl_default but not in sentiws and
# b. phrs which are only in sentiws but not in sepl_default and
# c. take the sentiment score from sentiws if both lists overlap

# case b.
df_sepl_withsentiws['opinion_value'] = np.where((df_sepl_withsentiws._merge == 'right_only'),
                                                df_sepl_withsentiws['sentiment'],
                                                df_sepl_withsentiws['opinion_value'])
# case c.
df_sepl_withsentiws['opinion_value'] = np.where((df_sepl_withsentiws._merge == 'both'),
                                                df_sepl_withsentiws['sentiment'],
                                                df_sepl_withsentiws['opinion_value'])
# subset columns
df_sepl_withsentiws = df_sepl_withsentiws[['phrase', 'opinion_value']]

# 3. Merge df from step 2 and step 1 and keep step 1 phrs
df_sentiment_final = pandas.merge(df_sepl_withsentiws, df_sepl_onlymodified,
                                  on='phrase',
                                  how='outer', indicator=True)
# keep only ...
# a. phrs which are only in df_sepl_withsentiws but not in df_sepl_onlymodified and
# b. phrs which are only in df_sepl_onlymodified but not in df_sepl_withsentiws and
# c. take the sentiment score from df_sepl_onlymodified if both lists overlap

# case b.
df_sentiment_final['opinion_value_x'] = np.where((df_sentiment_final._merge == 'right_only'),
                                                 df_sentiment_final['opinion_value_y'],
                                                 df_sentiment_final['opinion_value_x'])
# case c.
df_sentiment_final['opinion_value_x'] = np.where((df_sentiment_final._merge == 'both'),
                                                 df_sentiment_final['opinion_value_y'],
                                                 df_sentiment_final['opinion_value_x'])

# Finally, rename, subset columns and export
df_sentiment_final = df_sentiment_final[['phrase', 'opinion_value_x']].\
    rename(columns={'opinion_value_x': 'sentiment'})
df_sentiment_final.to_csv(path_data + '/Sentiment/SentimentList_final.csv', index=False)

# check
print(df_sentiment_final.describe(), '\n\ndone!')

###
