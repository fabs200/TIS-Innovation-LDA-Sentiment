import pandas, time
from python.ConfigUser import path_processedarticles
from python._AnalysisFunctions import LDACalibration, EstimateLDA
from python.params import params as p
import numpy as np

# Read in output file from PreprocessingSentences.py
complete_for_lda = pandas.read_csv(path_processedarticles + 'csv/complete_for_lda_{}_l.csv'.format(p['POStag_type']), sep='\t')

# complete_for_lda_calibr_tfidf = LDACalibration(
#                             type='tfidf',
#                             dataframecolumn=complete_for_lda[complete_for_lda['Art_unique'] == 1]['articles_{}_for_lda'.format(p['POStag_type'])],
#                             topics_start=1,
#                             topics_limit=10,
#                             topics_step=1,
#                             topn=25,
#                             num_words=25,
#                             metric='coherence',
#                             no_below=0.,
#                             no_above=1.,
#                             alpha='auto',
#                             eta='auto',
#                             eval_every=10,
#                             iterations=50,
#
#                             random_state=123,
#                             verbose=False,
#                             display_plot=True)


_metric = 'coherence'

for b in np.arange(.4, .5, .1):
    for a in np.arange(.8, .9, .1):

        results_lda_std = LDACalibration(
                                type='standard',
                                dataframecolumn=complete_for_lda[complete_for_lda['Art_unique'] == 1]['articles_{}_for_lda'.format(p['POStag_type'])],
                                # dataframecolumn=complete_for_lda['sentences_{}_for_lda'.format(p['POStag_type'])],
                                topics_start=1,
                                topics_limit=15,
                                topics_step=1,
                                topn=25,
                                num_words=25,
                                metric=_metric,
                                no_below=a,
                                no_above=b,
                                alpha='auto',
                                eta='auto',
                                eval_every=10,
                                iterations=50,

                                random_state=123,
                                verbose=False,
                                display_plot=True)

        print('############\n_metric: {}, no_below: {}, no_above: {},\n'.format(_metric, b, a), results_lda_std)

#
# ldamodel_sent_std, docsforlda_sent_std, ldadict_sent_std, ldacorpus_sent_std \
#         = EstimateLDA(complete_for_lda['sentences_{}_for_lda'.format(p['POStag_type'])],
#                       type='standard',
#                       no_below=p['no_below'],
#                       no_above=p['no_above'],
#                       num_topics=p['num_topics'],
#                       alpha=p['alpha'],
#                       eta=p['eta'],
#                       eval_every=p['eval_every'],
#                       iterations=p['iterations'],
#                       random_state=p['random_state'],
#                       verbose=p['verbose'],
#                       # further params
#                       distributed=p['distributed'], chunksize=p['chunksize'], callbacks=p['callbacks'],
#                       passes=p['passes'], update_every=p['update_every'], dtype=p['dtype'],
#                       decay=p['decay'], offset=p['offset'], gamma_threshold=p['gamma_threshold'],
#                       minimum_probability=p['minimum_probability'], ns_conf=p['ns_conf'],
#                       minimum_phi_value=p['minimum_phi_value'], per_word_topics=p['per_word_topics']
#                       )
#
#
# ldamodel_sent_tfidf, docsforlda_sent_tfidf, ldadict_sent_tfidf, ldacorpus_sent_tfidf \
#         = EstimateLDA(complete_for_lda['sentences_{}_for_lda'.format(p['POStag_type'])],
#                       type='tfidf',
#                       no_below=.0,
#                       no_above=1.,
#                       num_topics=p['num_topics'],
#                       alpha=p['alpha'],
#                       eta=p['eta'],
#                       eval_every=p['eval_every'],
#                       iterations=p['iterations'],
#                       random_state=p['random_state'],
#                       verbose=p['verbose'],
#                       # further params
#                       distributed=p['distributed'], chunksize=p['chunksize'], callbacks=p['callbacks'],
#                       passes=p['passes'], update_every=p['update_every'], dtype=p['dtype'],
#                       decay=p['decay'], offset=p['offset'], gamma_threshold=p['gamma_threshold'],
#                       minimum_probability=p['minimum_probability'], ns_conf=p['ns_conf'],
#                       minimum_phi_value=p['minimum_phi_value'], per_word_topics=p['per_word_topics']
#                       )
