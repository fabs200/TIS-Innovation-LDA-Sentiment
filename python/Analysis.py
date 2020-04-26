import pandas
from python.ConfigUser import path_processedarticles
from python.AnalysisFunctions import EstimateLDA, GetDomTopic

"""
---------------------
Analysis.py
---------------------
* Run estimation lda
* Extract dominant topic based
* Save long file
"""

# Specify POStag type
POStag_type = 'NN'

# Specify which level to run
level = 'sentence'
# level = ['sentence', 'paragraph', 'article']

# Specify on which level to fit lda
fitlevel = 'sentence'
# fitlevel = ['sentence', 'paragraph', 'article']

# Load long file
df_long = pandas.read_csv(path_processedarticles + 'csv/complete_for_lda_{}_l.csv'.format(POStag_type),
                          sep='\t', na_filter=False)

######
# TEMP keep first x articles
# df_long = df_long[df_long['Art_ID']<10]
######

"""
#########   LDA ESTIMATION  #########
"""

# lda on sentence level
if 'sentence' in level:
    ldamodel_sent, docsforlda_sent, ldadict_sent, ldacorpus_sent \
        = EstimateLDA(df_long['sentences_{}_for_lda'.format(POStag_type)])

# lda on paragraph level
if 'paragraph' in level:
    ldamodel_para, docsforlda_para, ldadict_para, ldacorpus_para \
        = EstimateLDA(df_long[df_long['Par_unique'] == 1]['paragraphs_{}_for_lda'.format(POStag_type)])

# lda on article level
if 'article' in level:
    ldamodel_arti, docsforlda_arti, ldadict_arti, ldacorpus_arti \
        = EstimateLDA(df_long[df_long['Art_unique'] == 1]['articles_{}_for_lda'.format(POStag_type)])


### Estimate Topic distribution of sentiment sentences and append to long file
## lda model based on sentences (*_sent_*)
# apply to sentences (*_sent_sent)
if 'sentence' in level and 'sentence' in fitlevel:
    df_long['DomTopic_sent_sent'] = \
        df_long['sentences_for_sentiment'].apply(lambda x: GetDomTopic(x, lda_model=ldamodel_sent,
                                                                       dict_lda=ldadict_sent))
# apply to paragraphs (*_sent_para)
if 'paragraph' in level and 'sentence' in fitlevel:
    df_long['DomTopic_sent_para'] = \
        df_long['paragraphs_text'].apply(lambda x: GetDomTopic(x, lda_model=ldamodel_sent,
                                                               dict_lda=ldadict_sent))
# apply to articles (*_sent_arti)
if 'article' in level and 'sentence' in fitlevel:
    df_long['DomTopic_sent_arti'] = \
        df_long['articles_text'].apply(lambda x: GetDomTopic(x, lda_model=ldamodel_sent,
                                                             dict_lda=ldadict_sent))

## lda model based on paragraphs (*_para_*)
# apply to sentences (*_para_sent)
if 'senctence' in level and 'paragraph' in fitlevel:
    df_long['DomTopic_para_sent'] = \
        df_long['sentences_for_sentiment'].apply(lambda x: GetDomTopic(x, lda_model=ldamodel_para,
                                                                       dict_lda=ldadict_para))
# apply to paragraphs (*_para_para)
if 'paragraph' in level and 'paragraph' in fitlevel:
    df_long['DomTopic_para_para'] = \
        df_long['paragraphs_text'].apply(lambda x: GetDomTopic(x, lda_model=ldamodel_para,
                                                               dict_lda=ldadict_para))
# apply to articles (*_para_arti)
if 'article' in level and 'paragraph' in fitlevel:
    df_long['DomTopic_para_arti'] = \
        df_long['articles_text'].apply(lambda x: GetDomTopic(x, lda_model=ldamodel_para,
                                                             dict_lda=ldadict_para))


## lda model based on articles (*_arti_*)
# apply to sentences (*_arti_sent)
if 'sentence' in level and 'article' in fitlevel:
    df_long['DomTopic_arti_sent'] = \
        df_long['sentences_for_sentiment'].apply(lambda x: GetDomTopic(x, lda_model=ldamodel_arti,
                                                                       dict_lda=ldadict_arti))
# apply to paragraphs (*_arti_para)
if 'paragraph' in level and 'article' in fitlevel:
    df_long['DomTopic_arti_para'] = \
        df_long['paragraphs_text'].apply(lambda x: GetDomTopic(x, lda_model=ldamodel_arti,
                                                               dict_lda=ldadict_arti))
# apply to articles (*_arti_arti)
if 'paragraph' in level and 'article' in fitlevel:
    df_long['DomTopic_arti_arti'] = \
        df_long['articles_text'].apply(lambda x: GetDomTopic(x, lda_model=ldamodel_arti,
                                                             dict_lda=ldadict_arti))


# Export file
df_long.to_csv(path_processedarticles + 'csv/lda_results_{}_l.csv'.format(POStag_type), sep='\t', index=False)
df_long.to_excel(path_processedarticles + 'lda_results_{}_l.xlsx'.format(POStag_type))


###
