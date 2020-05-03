import pandas, time
from python.ConfigUser import path_processedarticles
from python._AnalysisFunctions import LDACalibration, EstimateLDA
from python.params import params as p
import numpy as np

# Read in output file from PreprocessingSentences.py
complete_for_lda = pandas.read_csv(path_processedarticles + 'csv/complete_for_lda_{}_l.csv'.format(p['POStag_type']),
                                   sep='\t', na_filter=False)

# complete_for_lda_calibr_tfidf = LDACalibration(
#                             type='tfidf',
#                             dataframecolumn=complete_for_lda[complete_for_lda['Art_unique'] == 1]['articles_{}_for_lda'.format(p['POStag_type'])],
#                             topics_start=1,
#                             topics_limit=15,
#                             topics_step=1,
#                             topn=200,
#                             num_words=200,
#                             metric='hellinger',
#                             no_below=75,
#                             no_above=0.5,
#                             alpha='auto',
#                             eta='auto',
#                             eval_every=3,
#                             iterations=300,
#
#                             random_state=123,
#                             verbose=False,
#                             display_plot=True)


_metric = 'hellinger'

for no_b in np.arange(20, 101, 20):
    for no_a in np.arange(.3, .61, .1):
        # print(no_b, no_a)

        results_lda_std = LDACalibration(
                                type='tfidf',
                                dataframecolumn=complete_for_lda[complete_for_lda['Art_unique'] == 1]['articles_{}_for_lda'.format(p['POStag_type'])],
                                # dataframecolumn=complete_for_lda['sentences_{}_for_lda'.format(p['POStag_type'])],
                                topics_start=1,
                                topics_limit=15,
                                topics_step=1,
                                topn=250,
                                num_words=250,
                                metric=_metric,
                                no_below=no_b,
                                no_above=no_a,
                                alpha='auto',
                                eta='auto',
                                eval_every=5,
                                iterations=300,

                                random_state=123,
                                verbose=False,
                                display_plot=True)

        print('############\n_metric: {}, no_below: {}, no_above: {},\n'.format(_metric, no_b, no_a), results_lda_std)


ldamodel, docsforlda, ldadict, ldacorpus \
        = EstimateLDA(complete_for_lda[complete_for_lda['Art_unique'] == 1]['articles_{}_for_lda'.format(p['POStag_type'])],
                      type='tfidf',
                      no_below=p['no_below'],
                      no_above=p['no_above'],
                      num_topics=p['num_topics'],
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
                      minimum_phi_value=p['minimum_phi_value'], per_word_topics=p['per_word_topics']
                      )
