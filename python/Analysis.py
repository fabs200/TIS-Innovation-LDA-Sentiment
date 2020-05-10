import pandas, time
from python.ConfigUser import path_data
from python._AnalysisFunctions import EstimateLDA, GetDomTopic
from python.params import params as p

"""
---------------------
Analysis.py
---------------------
* Run estimation lda
* Extract dominant topics
* Save long file
"""

# unpack POStag type, lda_levels to run lda on, lda_level to get domtopic from
POStag, lda_level_fit, lda_level_domtopic = p['POStag'], p['lda_level_fit'], p['lda_level_domtopic']

# Load long file
print('Loading complete_for_lda_{}_l.csv'.format(POStag))
df_long = pandas.read_csv(path_data + 'csv/complete_for_lda_{}_l.csv'.format(POStag), sep='\t', na_filter=False)

######
# TEMP keep first x articles
# df_long = df_long[df_long['Art_ID']<100]
######

"""
#########   LDA ESTIMATION  #########
"""

start_time0 = time.process_time()

# lda on sentence lda_level
if 'sentence' in p['lda_level_fit']:
    print('\nestimating lda on sentence lda_level')
    start_ldafit_sent = time.process_time()
    ldamodel_sent, docsforlda_sent, ldadict_sent, ldacorpus_sent \
        = EstimateLDA(df_long['sentences_{}_for_lda'.format(POStag)],
                      type=p['type'],
                      no_below=p['no_below'],
                      no_above=p['no_above'],
                      num_topics=p['num_topics'],
                      num_words=p['num_words'],
                      alpha=p['alpha'],
                      eta=p['eta'],
                      eval_every=p['eval_every'],
                      iterations=p['iterations'],
                      random_state=p['random_state'],
                      verbose=p['verbose'],
                      # further params
                      distributed=p['distributed'], chunksize=p['chunksize'], callbacks=p['callbacks'],
                      passes=p['passes'], update_every=p['update_every'], dtype=p['dtype'],
                      decay=p['decay'], offset=p['offset'], gamma_threshold=p['gamma_threshold'],
                      minimum_probability=p['minimum_probability'], ns_conf=p['ns_conf'],
                      minimum_phi_value=p['minimum_phi_value'], per_word_topics=p['per_word_topics'],
                      # save model
                      save_model=True
                      )
    print('\testimating lda on sentence lda_level took {} seconds'.format(round(time.process_time()-start_ldafit_sent, 2)))


# lda on paragraph lda_level
if 'paragraph' in p['lda_level_fit']:
    print('\nestimating lda on paragraph lda_level')
    start_ldafit_para = time.process_time()
    ldamodel_para, docsforlda_para, ldadict_para, ldacorpus_para \
        = EstimateLDA(df_long[df_long['Par_unique'] == 1]['paragraphs_{}_for_lda'.format(POStag)],
                      type=p['type'],
                      no_below=p['no_below'],
                      no_above=p['no_above'],
                      num_topics=p['num_topics'],
                      num_words=p['num_words'],
                      alpha=p['alpha'],
                      eta=p['eta'],
                      eval_every=p['eval_every'],
                      iterations=p['iterations'],
                      random_state=p['random_state'],
                      verbose=p['verbose'],
                      # further params
                      distributed=p['distributed'], chunksize=p['chunksize'], callbacks=p['callbacks'],
                      passes=p['passes'], update_every=p['update_every'], dtype=p['dtype'],
                      decay=p['decay'], offset=p['offset'], gamma_threshold=p['gamma_threshold'],
                      minimum_probability=p['minimum_probability'], ns_conf=p['ns_conf'],
                      minimum_phi_value=p['minimum_phi_value'], per_word_topics=p['per_word_topics'],
                      # save model
                      save_model=True
                      )
    print('\testimating lda on paragraph lda_level took {} seconds'.format(round(time.process_time()-start_ldafit_para, 2)))

# lda on article lda_level
if 'article' in p['lda_level_fit']:
    print('\nestimating lda on article lda_level')
    start_ldafit_arti = time.process_time()
    ldamodel_arti, docsforlda_arti, ldadict_arti, ldacorpus_arti \
        = EstimateLDA(df_long[df_long['Art_unique'] == 1]['articles_{}_for_lda'.format(POStag)],
                      type=p['type'],
                      no_below=p['no_below'],
                      no_above=p['no_above'],
                      num_topics=p['num_topics'],
                      num_words=p['num_words'],
                      alpha=p['alpha'],
                      eta=p['eta'],
                      eval_every=p['eval_every'],
                      iterations=p['iterations'],
                      random_state=p['random_state'],
                      verbose=p['verbose'],
                      # further params
                      distributed=p['distributed'], chunksize=p['chunksize'], callbacks=p['callbacks'],
                      passes=p['passes'], update_every=p['update_every'], dtype=p['dtype'],
                      decay=p['decay'], offset=p['offset'], gamma_threshold=p['gamma_threshold'],
                      minimum_probability=p['minimum_probability'], ns_conf=p['ns_conf'],
                      minimum_phi_value=p['minimum_phi_value'], per_word_topics=p['per_word_topics'],
                      # save model
                      save_model=True
                      )
    print('\testimating lda on article lda_level took {} seconds'.format(round(time.process_time()-start_ldafit_arti, 2)))

end_time0 = time.process_time()
print('\ttimer0: Elapsed time is {} seconds'.format(round(end_time0-start_time0, 2)))

start_time1 = time.process_time()


######
# TEMP keep first x articles
# df_long = df_long[df_long['Art_ID']<101]
######


### Estimate Topic distribution of sentiment sentences and append to long file
## lda model based on sentences (*_sent_*)
# apply to sentences (*_sent_sent)
if 'sentence' in p['lda_level_domtopic'] and 'sentence' in p['lda_level_fit']:
    print('\nretrieving dominant topics from sentences (estimated lda on sentence)')
    start_domtopic_sent_sent = time.process_time()
    df_long['DomTopic_sent_sent'] = \
        df_long.apply(lambda row: GetDomTopic(row['sentences_for_sentiment'],
                                              lda_model=ldamodel_sent, dict_lda=ldadict_sent), axis=1)
    df_long['DomTopic_sent_sent_id'] = df_long['DomTopic_sent_sent'].str[0]
    df_long['DomTopic_sent_sent_prob'] = df_long['DomTopic_sent_sent'].str[1]
    print('\tretrieving dominant topics from sentences took {} seconds,'.format(
        round(time.process_time()-start_domtopic_sent_sent, 2)))

# apply to paragraphs (*_sent_para)
if 'paragraph' in p['lda_level_domtopic'] and 'sentence' in p['lda_level_fit']:
    print('\nretrieving dominant topics from paragraphs (estimated lda on sentence)')
    start_domtopic_sent_para = time.process_time()
    df_long['DomTopic_sent_para'] = \
        df_long.apply(lambda row: GetDomTopic(row['paragraphs_text'],
                                              lda_model=ldamodel_sent, dict_lda=ldadict_sent), axis=1)
    df_long['DomTopic_sent_para_id'] = df_long['DomTopic_sent_para'].str[0]
    df_long['DomTopic_sent_para_prob'] = df_long['DomTopic_sent_para'].str[1]
    print('\tretrieving dominant topics from paragraph took {} seconds'.format(
        round(time.process_time()-start_domtopic_sent_para, 2)))

# apply to articles (*_sent_arti)
if 'article' in p['lda_level_domtopic'] and 'sentence' in p['lda_level_fit']:
    print('\nretrieving dominant topics from articles (estimated lda on sentence)')
    start_domtopic_sent_arti = time.process_time()
    df_long['DomTopic_sent_arti'] = \
        df_long.apply(lambda row: GetDomTopic(row['articles_text'],
                                              lda_model=ldamodel_sent, dict_lda=ldadict_sent), axis=1)
    df_long['DomTopic_sent_arti_id'] = df_long['DomTopic_sent_arti'].str[0]
    df_long['DomTopic_sent_arti_prob'] = df_long['DomTopic_sent_arti'].str[1]
    print('\tretrieving dominant topics from articles took {} seconds'.format(
        round(time.process_time()-start_domtopic_sent_arti, 2)))

## lda model based on paragraphs (*_para_*)
# apply to sentences (*_para_sent)
if 'senctence' in p['lda_level_domtopic'] and 'paragraph' in p['lda_level_fit']:
    print('\nretrieving dominant topics from sentences (estimated lda on paragraph)')
    start_domtopic_para_sent = time.process_time()
    df_long['DomTopic_para_sent'] = \
        df_long.apply(lambda row: GetDomTopic(row['sentences_for_sentiment'],
                                              lda_model=ldamodel_para, dict_lda=ldadict_para), axis=1)
    df_long['DomTopic_para_sent_id'] = df_long['DomTopic_para_sent'].str[0]
    df_long['DomTopic_para_sent_prob'] = df_long['DomTopic_para_sent'].str[1]
    print('\tretrieving dominant topics from sentences took {} seconds'.format(
        round(time.process_time()-start_domtopic_para_sent, 2)))

# apply to paragraphs (*_para_para)
if 'paragraph' in p['lda_level_domtopic'] and 'paragraph' in p['lda_level_fit']:
    print('\nretrieving dominant topics from paragraph (estimated lda on paragraph)')
    start_domtopic_para_para = time.process_time()
    df_long['DomTopic_para_para'] = \
        df_long.apply(lambda row: GetDomTopic(row['paragraphs_text'],
                                              lda_model=ldamodel_para, dict_lda=ldadict_para), axis=1)
    df_long['DomTopic_para_para_id'] = df_long['DomTopic_para_para'].str[0]
    df_long['DomTopic_para_para_prob'] = df_long['DomTopic_para_para'].str[1]
    print('\tretrieving dominant topics from paragraphs took {} seconds'.format(
        round(time.process_time()-start_domtopic_para_para, 2)))

# apply to articles (*_para_arti)
if 'article' in p['lda_level_domtopic'] and 'paragraph' in p['lda_level_fit']:
    print('\nretrieving dominant topics from article (estimated lda on paragraph)')
    start_domtopic_para_arti = time.process_time()
    df_long['DomTopic_para_arti'] = \
        df_long.apply(lambda row: GetDomTopic(row['articles_text'],
                                              lda_model=ldamodel_para, dict_lda=ldadict_para), axis=1)
    df_long['DomTopic_para_arti_id'] = df_long['DomTopic_para_arti'].str[0]
    df_long['DomTopic_para_arti_prob'] = df_long['DomTopic_para_arti'].str[1]
    print('\tretrieving dominant topics from articles took {} seconds'.format(
        round(time.process_time()-start_domtopic_para_arti, 2)))

## lda model based on articles (*_arti_*)
# apply to sentences (*_arti_sent)
if 'sentence' in p['lda_level_domtopic'] and 'article' in p['lda_level_fit']:
    print('\nretrieving dominant topics from sentences (estimated lda on article)')
    start_domtopic_arti_sent = time.process_time()
    df_long['DomTopic_arti_sent'] = \
        df_long.apply(lambda row: GetDomTopic(row['sentences_for_sentiment'],
                                              lda_model=ldamodel_arti, dict_lda=ldadict_arti), axis=1)
    df_long['DomTopic_arti_sent_id'] = df_long['DomTopic_arti_sent'].str[0]
    df_long['DomTopic_arti_sent_prob'] = df_long['DomTopic_arti_sent'].str[1]
    print('\tretrieving dominant topics from sentences took {} seconds'.format(
        round(time.process_time()-start_domtopic_arti_sent, 2)))

# apply to paragraphs (*_arti_para)
if 'paragraph' in p['lda_level_domtopic'] and 'article' in p['lda_level_fit']:
    print('\nretrieving dominant topics from paragraphs (estimated lda on article)')
    start_domtopic_arti_para = time.process_time()
    df_long['DomTopic_arti_para'] = \
        df_long.apply(lambda row: GetDomTopic(row['paragraphs_text'],
                                              lda_model=ldamodel_arti, dict_lda=ldadict_arti), axis=1)
    df_long['DomTopic_arti_para_id'] = df_long['DomTopic_arti_para'].str[0]
    df_long['DomTopic_arti_para_prob'] = df_long['DomTopic_arti_para'].str[1]
    print('\tretrieving dominant topics from paragraphs took {} seconds'.format(
        round(time.process_time()-start_domtopic_arti_para, 2)))

# apply to articles (*_arti_arti)
if 'article' in p['lda_level_domtopic'] and 'article' in p['lda_level_fit']:
    print('\nretrieving dominant topics from articles (estimated lda on article)')
    start_domtopic_arti_arti = time.process_time()
    df_long['DomTopic_arti_arti'] = \
        df_long.apply(lambda row:
                      GetDomTopic(row['articles_text'], lda_model=ldamodel_arti, dict_lda=ldadict_arti)
                      if row['Art_unique']==1 else '', axis=1)
    df_long['DomTopic_arti_arti_id'] = df_long['DomTopic_arti_arti'].str[0]
    df_long['DomTopic_arti_arti_prob'] = df_long['DomTopic_arti_arti'].str[1]

    print('\tretrieving dominant topics from articles took {} seconds'.format(
        round(time.process_time()-start_domtopic_arti_arti, 2)))

end_time1 = time.process_time()
print('\ttimer1: Elapsed time is {} seconds'.format(round(end_time1-start_time1, 2)))

# Export file
print('\nsaving results')
df_long.to_csv(path_data + 'csv/lda_results_{}_l.csv'.format(p['currmodel']), sep='\t', index=False)
df_long.to_excel(path_data + 'lda_results_{}_l.xlsx'.format(p['currmodel']))

print('Overall elapsed time:', round(time.process_time()-start_time0, 2))

###
