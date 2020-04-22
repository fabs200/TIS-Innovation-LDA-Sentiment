import pandas, time, dtale
from python.ConfigUser import path_processedarticles
import dtale
from python.ProcessingFunctions import Sentencizer

# Specify POStag type
POStag_type = 'NN'

start_time0 = time.process_time()
print('loading files')

### Read in long files
# Sentence long file
df_long_sent = pandas.read_csv(path_processedarticles + 'csv/sentences_for_lda_{}_l.csv'.format(POStag_type), sep='\t')

# Paragraph long file
df_long_para = pandas.read_csv(path_processedarticles + 'csv/paragraphs_for_lda_{}_l.csv'.format(POStag_type), sep='\t')

# Article long file
df_long_arti = pandas.read_csv(path_processedarticles + 'csv/articles_for_lda_{}_l.csv'.format(POStag_type), sep='\t')

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
df_long_para['parag_by_sents'] = df_long_para['paragraphs_for_sentiment'].apply(lambda x: Sentencizer(x))

end_time1 = time.process_time()
print('\ttimer1: Elapsed time is {} seconds.'.format(round(end_time1-start_time1, 2)))

start_time2 = time.process_time()

print('explode(), Sent_ID, sort cols')

### 2. Make Long-file by df_long_para['parag_by_sents']
df_long_para_by_sents = df_long_para.explode('parag_by_sents')

### 3. Generate Sent_ID as in df_long_sent
df_long_para_by_sents['Sent_ID'] = df_long_para_by_sents.groupby(['Art_ID']).cumcount()+1

### 4. Sort columns
df_long_para_by_sents = df_long_para_by_sents[['Art_ID', 'Par_ID', 'Sent_ID', 'Newspaper', 'Date',
                                               'paragraphs_for_sentiment', 'paragraphs_{}_for_lda'.format(POStag_type)]]
# dtale.show(df_long_para_by_sents, ignore_duplicate=True)

end_time2 = time.process_time()
print('\ttimer2: Elapsed time is {} seconds.'.format(round(end_time2-start_time2, 2)))

start_time3 = time.process_time()

print('Merge long-files and process Date')

### 5. Merge df_long_sent and df_long_arti to df_long_para_by_sents
df_long_complete = pandas.merge(df_long_para_by_sents, df_long_sent[['Art_ID', 'Sent_ID', 'sentences_for_sentiment',
                                                                     'sentences_{}_for_lda'.format(POStag_type)]],
                                how='inner', on=['Art_ID', 'Sent_ID'],
                                suffixes=('_para', '_sent'),
                                indicator=True, validate='1:1')\
    .rename(columns={'_merge': '_merge_sent2para'})\
    \
    .merge(df_long_arti[['Art_ID', 'articles_for_sentiment', 'articles_{}_for_lda'.format(POStag_type)]],
           how='inner', on=['Art_ID'],
           suffixes=(None, None),
           indicator=True, validate='m:1')\
    .rename(columns={'_merge': '_merge_arti2para'})

# Check
# dtale.show(df_long_complete, ignore_duplicate=True)

### 6. Process Date variable
df_long_complete['Date'] = pandas.to_datetime(df_long_complete['Date'], format='%Y-%m-%d')
df_long_complete['month'] = df_long_complete['Date'].dt.to_period('M')
df_long_complete['year'] = df_long_complete['Date'].dt.to_period('Y')

### 7. Order vars
df_long_complete = df_long_complete[['Art_ID', 'Par_ID', 'Sent_ID', 'year', 'month', 'Newspaper',
                                     'sentences_for_sentiment', 'sentences_{}_for_lda'.format(POStag_type),
                                     'paragraphs_for_sentiment', 'paragraphs_{}_for_lda'.format(POStag_type),
                                     'articles_for_sentiment', 'articles_{}_for_lda'.format(POStag_type)]]

end_time3 = time.process_time()
print('\ttimer3: Elapsed time is {} seconds.'.format(round(end_time3-start_time3, 2)))

start_time4 = time.process_time()

### 8. Export longfile to csv (will be read in later)
print('Save final df_long_complete')
df_long_complete.to_csv(path_processedarticles + 'csv/long_compl_for_lda_{}_l.csv'.format(POStag_type),
                        sep='\t', index=False)
df_long_complete.to_excel(path_processedarticles + 'long_compl_for_lda_{}_l.xlsx'.format(POStag_type))

end_time4 = time.process_time()
print('\ttimer4: Elapsed time is {} seconds.'.format(round(end_time4-start_time4, 2)), end='\r')
print('\nOverall elapsed time is {} seconds.'.format(round(end_time4-start_time0, 2)))

###
