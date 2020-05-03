### Update Long file only by newly preprocessed articles_NN_for_lda
import pandas
from python.ConfigUser import path_processedarticles
from python.params import params as p

# unpack POStag type
POStag_type = p['POStag_type']

# Read in article long file
df_long_arti = pandas.read_csv(path_processedarticles + 'csv/articles_for_lda_{}_l.csv'.format(POStag_type),
                               sep='\t', na_filter=False)

# Read in exported long file from ProcessLongfiles.py
df_long_complete = pandas.read_csv(path_processedarticles + 'csv/complete_for_lda_{}_l.csv'.format(POStag_type),
                                   sep='\t', na_filter=False)

# Drop old column
df_long_complete = df_long_complete.drop(columns=['articles_{}_for_lda'.format(POStag_type),
                                                  'articles_text',
                                                  '_merge'])

# Update column from df_long_arti
df_long_complete = pandas.merge(df_long_complete,
                                df_long_arti[['Art_ID', 'articles_{}_for_lda'.format(POStag_type), 'articles_text']],
                                how='inner', on=['Art_ID'], indicator=True)

### Export longfile to csv (will be read in later)
print('Save updated final df_long_complete')
df_long_complete.to_csv(path_processedarticles + 'csv/complete_for_lda_{}_l.csv'.format(POStag_type),
                        sep='\t', index=False)
df_long_complete.to_excel(path_processedarticles + 'complete_for_lda_{}_l.xlsx'.format(POStag_type))

###
