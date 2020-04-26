import pandas, time
from python.ConfigUser import path_processedarticles
from python.AnalysisFunctions import EstimateLDA, GetDomTopic

"""
---------------------
Analysis.py
---------------------
* Run estimation lda
* Extract dominant topics
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
print('Loading complete_for_lda_{}_l.csv'.format(POStag_type))
df_long = pandas.read_csv(path_processedarticles + 'csv/complete_for_lda_{}_l.csv'.format(POStag_type),
                          sep='\t', na_filter=False)

######
# TEMP keep first x articles
# df_long = df_long[df_long['Art_ID']<10]
######

"""
#########   LDA ESTIMATION  #########
"""

start_time0 = time.process_time()

# lda on sentence level
if 'sentence' in level:
    print('estimating lda on sentence level')
    start_ldafit_sent = time.process_time()
    ldamodel_sent, docsforlda_sent, ldadict_sent, ldacorpus_sent \
        = EstimateLDA(df_long['sentences_{}_for_lda'.format(POStag_type)])
    print('\testimating lda on sentence level took {} seconds'.format(round(time.process_time()-start_ldafit_sent, 2)))


# lda on paragraph level
if 'paragraph' in level:
    print('estimating lda on paragraph level')
    start_ldafit_para = time.process_time()
    ldamodel_para, docsforlda_para, ldadict_para, ldacorpus_para \
        = EstimateLDA(df_long[df_long['Par_unique'] == 1]['paragraphs_{}_for_lda'.format(POStag_type)])
    print('\testimating lda on paragraph level took {} seconds'.format(round(time.process_time()-start_ldafit_para, 2)))

# lda on article level
if 'article' in level:
    print('estimating lda on article level')
    start_ldafit_arti = time.process_time()
    ldamodel_arti, docsforlda_arti, ldadict_arti, ldacorpus_arti \
        = EstimateLDA(df_long[df_long['Art_unique'] == 1]['articles_{}_for_lda'.format(POStag_type)])
    print('\testimating lda on article level took {} seconds'.format(round(time.process_time()-start_ldafit_arti, 2)))

end_time0 = time.process_time()
print('\ttimer0: Elapsed time is {} seconds'.format(round(end_time0-start_time0, 2)))

start_time1 = time.process_time()


### Estimate Topic distribution of sentiment sentences and append to long file
## lda model based on sentences (*_sent_*)
# apply to sentences (*_sent_sent)
if 'sentence' in level and 'sentence' in fitlevel:
    print('retrieving dominant topics from sentences (estimated lda on sentence)')
    start_domtopic_sent_sent = time.process_time()
    df_long['DomTopic_sent_sent'] = \
        df_long['sentences_for_sentiment'].apply(lambda x: GetDomTopic(x, lda_model=ldamodel_sent, dict_lda=ldadict_sent))
    print('\tretrieving dominant topics from sentences took {} seconds,'.format(
        round(time.process_time()-start_domtopic_sent_sent, 2)))

# apply to paragraphs (*_sent_para)
if 'paragraph' in level and 'sentence' in fitlevel:
    print('retrieving dominant topics from paragraphs (estimated lda on sentence)')
    start_domtopic_sent_para = time.process_time()
    df_long['DomTopic_sent_para'] = \
        df_long['paragraphs_text'].apply(lambda x: GetDomTopic(x, lda_model=ldamodel_sent, dict_lda=ldadict_sent))
    print('\tretrieving dominant topics from paragraph took {} seconds'.format(
        round(time.process_time()-start_domtopic_sent_para, 2)))

# apply to articles (*_sent_arti)
if 'article' in level and 'sentence' in fitlevel:
    print('retrieving dominant topics from articles (estimated lda on sentence)')
    start_domtopic_sent_arti = time.process_time()
    df_long['DomTopic_sent_arti'] = \
        df_long['articles_text'].apply(lambda x: GetDomTopic(x, lda_model=ldamodel_sent, dict_lda=ldadict_sent))
    print('\tretrieving dominant topics from articles took {} seconds'.format(
        round(time.process_time()-start_domtopic_sent_arti, 2)))

## lda model based on paragraphs (*_para_*)
# apply to sentences (*_para_sent)
if 'senctence' in level and 'paragraph' in fitlevel:
    print('retrieving dominant topics from sentences (estimated lda on paragraph)')
    start_domtopic_para_sent = time.process_time()
    df_long['DomTopic_para_sent'] = \
        df_long['sentences_for_sentiment'].apply(lambda x: GetDomTopic(x, lda_model=ldamodel_para, dict_lda=ldadict_para))
    print('\tretrieving dominant topics from sentences took {} seconds'.format(
        round(time.process_time()-start_domtopic_para_sent, 2)))

# apply to paragraphs (*_para_para)
if 'paragraph' in level and 'paragraph' in fitlevel:
    print('retrieving dominant topics from paragraph (estimated lda on paragraph)')
    start_domtopic_para_para = time.process_time()
    df_long['DomTopic_para_para'] = \
        df_long['paragraphs_text'].apply(lambda x: GetDomTopic(x, lda_model=ldamodel_para, dict_lda=ldadict_para))
    print('\tretrieving dominant topics from paragraphs took {} seconds'.format(
        round(time.process_time()-start_domtopic_para_para, 2)))

# apply to articles (*_para_arti)
if 'article' in level and 'paragraph' in fitlevel:
    print('retrieving dominant topics from article (estimated lda on paragraph)')
    start_domtopic_para_arti = time.process_time()
    df_long['DomTopic_para_arti'] = \
        df_long['articles_text'].apply(lambda x: GetDomTopic(x, lda_model=ldamodel_para, dict_lda=ldadict_para))
    print('\tretrieving dominant topics from articles took {} seconds'.format(
        round(time.process_time()-start_domtopic_para_arti, 2)))

## lda model based on articles (*_arti_*)
# apply to sentences (*_arti_sent)
if 'sentence' in level and 'article' in fitlevel:
    print('retrieving dominant topics from sentences (estimated lda on article)')
    start_domtopic_arti_sent = time.process_time()
    df_long['DomTopic_arti_sent'] = \
        df_long['sentences_for_sentiment'].apply(lambda x: GetDomTopic(x, lda_model=ldamodel_arti, dict_lda=ldadict_arti))
    print('\tretrieving dominant topics from sentences took {} seconds'.format(
        round(time.process_time()-start_domtopic_arti_sent, 2)))

# apply to paragraphs (*_arti_para)
if 'paragraph' in level and 'article' in fitlevel:
    print('retrieving dominant topics from paragraphs (estimated lda on article)')
    start_domtopic_arti_para = time.process_time()
    df_long['DomTopic_arti_para'] = \
        df_long['paragraphs_text'].apply(lambda x: GetDomTopic(x, lda_model=ldamodel_arti, dict_lda=ldadict_arti))
    print('\tretrieving dominant topics from paragraphs took {} seconds'.format(
        round(time.process_time()-start_domtopic_arti_para, 2)))

# apply to articles (*_arti_arti)
if 'paragraph' in level and 'article' in fitlevel:
    print('retrieving dominant topics from articles (estimated lda on article)')
    start_domtopic_arti_arti = time.process_time()
    df_long['DomTopic_arti_arti'] = \
        df_long['articles_text'].apply(lambda x: GetDomTopic(x, lda_model=ldamodel_arti, dict_lda=ldadict_arti))
    print('\tretrieving dominant topics from articles took {} seconds'.format(
        round(time.process_time()-start_domtopic_arti_arti, 2)))

end_time1 = time.process_time()
print('\ttimer1: Elapsed time is {} seconds'.format(round(end_time1-start_time1, 2)))


# Export file
print('saving results')
df_long.to_csv(path_processedarticles + 'csv/lda_results_{}_l.csv'.format(POStag_type), sep='\t', index=False)
df_long.to_excel(path_processedarticles + 'lda_results_{}_l.xlsx'.format(POStag_type))

print('Overall elapsed time:', round(time.process_time()-start_time0, 2))

###
