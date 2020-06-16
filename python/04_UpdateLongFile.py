### Update Long file only by newly preprocessed articles_NN_for_lda
import pandas
from python.ConfigUser import path_data
from python.params import params as p

# unpack POStag type
POStag_type = p['POStag']

# Read in article long file
df_long_arti = pandas.read_csv(path_data + 'csv/articles_for_lda_{}_l.csv'.format(POStag_type),
                               sep='\t', na_filter=False)
# Read in paragraph long file
df_long_para = pandas.read_csv(path_data + 'csv/paragraphs_for_lda_{}_l.csv'.format(POStag_type),
                               sep='\t', na_filter=False)
# Read in sentence long file
df_long_sent = pandas.read_csv(path_data + 'csv/sentences_for_lda_{}_l.csv'.format(POStag_type),
                               sep='\t', na_filter=False)

# Read in exported long file from 04_ProcessLongfiles.py
df_long_complete = pandas.read_csv(path_data + 'csv/complete_for_lda_{}_l.csv'.format(POStag_type),
                                   sep='\t', na_filter=False)

# Drop old column
df_long_complete = df_long_complete.drop(columns=['articles_{}_for_lda'.format(POStag_type),
                                                  'paragraphs_{}_for_lda'.format(POStag_type),
                                                  'sentences_{}_for_lda'.format(POStag_type),
                                                  'articles_text',
                                                  'paragraphs_text',
                                                  'sentences_for_sentiment'
                                                  ])

# Update column from df_long_arti and rename _merge
df_long_complete = pandas.merge(df_long_complete,
                                df_long_arti[['Art_ID',
                                              'articles_{}_for_lda'.format(POStag_type), 'articles_text']],
                                how='inner', on=['Art_ID'], indicator=True).rename(columns={'_merge': '_merge_arti'})

# Update column from df_long_para and rename _merge
df_long_complete = pandas.merge(df_long_complete,
                                df_long_para[['Art_ID', 'Par_ID',
                                              'paragraphs_{}_for_lda'.format(POStag_type), 'paragraphs_text']],
                                how='inner', on=['Art_ID', 'Par_ID'], indicator=True).rename(columns={'_merge': '_merge_para'})

# Update column from df_long_sent and rename _merge
df_long_complete = pandas.merge(df_long_complete,
                                df_long_sent[['Art_ID', 'Sent_ID',
                                              'sentences_{}_for_lda'.format(POStag_type), 'sentences_for_sentiment']],
                                how='inner', on=['Art_ID', 'Sent_ID'], indicator=True).rename(columns={'_merge': '_merge_sent'})

# check merge
assert df_long_complete['_merge_arti'].unique() == \
       df_long_complete['_merge_para'].unique() == \
       df_long_complete['_merge_sent'].unique() == ['both']
df_long_complete = df_long_complete.drop(columns=['_merge_arti', '_merge_para', '_merge_sent'])

### Export longfile to csv (will be read in later)
print('Save updated final df_long_complete')
df_long_complete.to_csv(path_data + 'csv/complete_for_lda_{}_l.csv'.format(POStag_type),
                        sep='\t', index=False)
df_long_complete.to_excel(path_data + 'complete_for_lda_{}_l.xlsx'.format(POStag_type))

###
