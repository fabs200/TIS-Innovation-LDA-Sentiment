import pandas, time, dtale
from python.ConfigUser import path_processedarticles
from python.ProcessingFunctions import Sentencizer, IgnoreWarnings
from python.AnalysisFunctions import Load_SePL, GetSentimentScores_long

"""
---------------------
ProcessLongfiles.py
---------------------
* Load all Long files and put together
* Produce Par_ID, Sent_ID, Art_ID variables
* prepare date vars
* extract sentiment score
* Save long-file
"""

# Specify POStag type
POStag_type = 'NN'

start_time0 = time.process_time()
print('loading files')

### Read in long files
# Sentence long file
df_long_sent = pandas.read_csv(path_processedarticles + 'csv/sentences_for_lda_{}_l.csv'.format(POStag_type),
                               sep='\t', na_filter=False)

# Paragraph long file
df_long_para = pandas.read_csv(path_processedarticles + 'csv/paragraphs_for_lda_{}_l.csv'.format(POStag_type),
                               sep='\t', na_filter=False)

# Article long file
df_long_arti = pandas.read_csv(path_processedarticles + 'csv/articles_for_lda_{}_l.csv'.format(POStag_type),
                               sep='\t', na_filter=False)

######
# TEMP: Make smaller
# df_long_sent = df_long_sent[df_long_sent['Art_ID']<10]
# df_long_para = df_long_para[df_long_para['Art_ID']<10]
######

end_time0 = time.process_time()
print('\ttimer0: Elapsed time is {} seconds.'.format(round(end_time0-start_time0, 2)))

start_time1 = time.process_time()

### 1. Split paragraphs by sents via same fct Sentencizer() as in PreprocessingSentences
print('Sentencizer()')
df_long_para['parag_by_sents'] = df_long_para['paragraphs_text'].apply(lambda x: Sentencizer(x))

end_time1 = time.process_time()
print('\ttimer1: Elapsed time is {} seconds.'.format(round(end_time1-start_time1, 2)))

start_time2 = time.process_time()

print('explode(), Sent_ID, sort cols')

### 2. Make Long-file by df_long_para['parag_by_sents']
df_long_para_by_sents = df_long_para.explode('parag_by_sents')

### 3. Generate Sent_ID as in df_long_sent
df_long_para_by_sents['Sent_ID'] = df_long_para_by_sents.groupby(['Art_ID']).cumcount()+1

### 4. Generate dummies for articles, paragraphs to identify unique observations (prevent double counting)
df_long_para_by_sents['Art_unique'] = [1 if x==1 else 0 for x in df_long_para_by_sents['Sent_ID']]
df_long_para_by_sents['Par_unique'] = [1 if x==1 else 0 for x in df_long_para_by_sents.groupby(['Art_ID', 'Par_ID']).cumcount()+1]

### 5. Sort columns
df_long_para_by_sents = df_long_para_by_sents[['Art_ID', 'Par_ID', 'Sent_ID', 'Art_unique', 'Par_unique',
                                               'Newspaper', 'Date',
                                               'paragraphs_text', 'paragraphs_{}_for_lda'.format(POStag_type)]]
# dtale.show(df_long_para_by_sents, ignore_duplicate=True)

end_time2 = time.process_time()
print('\ttimer2: Elapsed time is {} seconds.'.format(round(end_time2-start_time2, 2)))

start_time3 = time.process_time()

print('Merge long-files and process Date')

### 6. Merge df_long_sent and df_long_arti to df_long_para_by_sents
df_long_complete = pandas.merge(df_long_para_by_sents, df_long_sent[['Art_ID', 'Sent_ID', 'sentences_for_sentiment',
                                                                     'sentences_{}_for_lda'.format(POStag_type)]],
                                how='inner', on=['Art_ID', 'Sent_ID'],
                                suffixes=('_para', '_sent'),
                                indicator=True, validate='1:1')\
    .rename(columns={'_merge': '_merge_sent2para'})\
    \
    .merge(df_long_arti[['Art_ID', 'articles_text', 'articles_{}_for_lda'.format(POStag_type)]],
           how='inner', on=['Art_ID'],
           suffixes=(None, None),
           indicator=True, validate='m:1')\
    .rename(columns={'_merge': '_merge_arti2para'})

# Check
# dtale.show(df_long_complete, ignore_duplicate=True)

### 7. Process Date variable
df_long_complete['Date'] = pandas.to_datetime(df_long_complete['Date'], format='%Y-%m-%d')
df_long_complete['month'] = df_long_complete['Date'].dt.to_period('M')
df_long_complete['quarter'] = df_long_complete['Date'].dt.quarter
df_long_complete['year'] = df_long_complete['Date'].dt.to_period('Y')

### 8. Create unique Articles, Paragraphs (prevent double counting)
# df_long_complete['paragraphs_text'] = np.where(df_long_complete['Par_unique']==1, df_long_complete['paragraphs_text'], '')
# df_long_complete['articles_text'] = np.where(df_long_complete['Art_unique']==1, df_long_complete['articles_text'], '')
# df_long_complete['paragraphs_{}_for_lda'.format(POStag_type)] = np.where(df_long_complete['Par_unique']==1, df_long_complete['paragraphs_{}_for_lda'.format(POStag_type)], '')
# df_long_complete['articles_{}_for_lda'.format(POStag_type)] = np.where(df_long_complete['Art_unique']==1, df_long_complete['articles_{}_for_lda'.format(POStag_type)], '')

# dtale.show(df_long_complete, ignore_duplicate=True)

end_time3 = time.process_time()
print('\ttimer3: Elapsed time is {} seconds.'.format(round(end_time3-start_time3, 2)))

start_time4 = time.process_time()


### 9. Extract Sentiment Score

# Read in SePL
df_sepl = Load_SePL()

# Extract sentiment score
print('Extract sentiment scores')
IgnoreWarnings()
df_long_complete['Sentiment_score_dict'] = \
    df_long_complete['sentences_for_sentiment'].apply(lambda x: GetSentimentScores_long(sent=x, df_sepl=df_sepl))

# Split columns from 'Sentiment_score_temp' and concatenate df_temp
df_long_complete = \
    pandas.concat([df_long_complete, pandas.json_normalize(df_long_complete['Sentiment_score_dict'])], axis=1)

end_time4 = time.process_time()
print('\ttimer4: Elapsed time is {} seconds.'.format(round(end_time4-start_time4, 2)))

start_time5 = time.process_time()

### 10. Rename and order vars
print('Order vars')
df_long_complete = df_long_complete.rename(columns={'mean': 'sentiscore_mean',
                                                    'median': 'sentiscore_median',
                                                    'n': 'sentiscore_n',
                                                    'sd': 'sentiscore_sd',
                                                    'SentiScores': 'sentiscore_scores',
                                                    'seplphrs': 'sentiscore_seplphrs'})

df_long_complete = df_long_complete[[
    # IDs
    'Art_ID', 'Par_ID', 'Sent_ID', 'Art_unique', 'Par_unique',
    # article infos
    'year', 'quarter', 'month', 'Newspaper',
    # text
    'sentences_for_sentiment', 'sentences_{}_for_lda'.format(POStag_type),
    'paragraphs_text', 'paragraphs_{}_for_lda'.format(POStag_type),
    'articles_text', 'articles_{}_for_lda'.format(POStag_type),
    # Sentiment Scores
    'sentiscore_mean', 'sentiscore_median', 'sentiscore_n', 'sentiscore_sd', 'sentiscore_scores', 'sentiscore_seplphrs'
]]

### 11. Export longfile to csv (will be read in later)
print('Save final df_long_complete')
df_long_complete.to_csv(path_processedarticles + 'csv/complete_for_lda_{}_l.csv'.format(POStag_type),
                        sep='\t', index=False)
df_long_complete.to_excel(path_processedarticles + 'complete_for_lda_{}_l.xlsx'.format(POStag_type))

end_time5 = time.process_time()
print('\ttimer5: Elapsed time is {} seconds.'.format(round(end_time5-start_time5, 2)))
print('\nOverall elapsed time is {} seconds.'.format(round(end_time5-start_time0, 2)))

###