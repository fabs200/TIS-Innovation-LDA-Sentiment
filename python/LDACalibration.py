import pandas, time
from python.ConfigUser import path_data
from python._AnalysisFunctions import LDACalibration, EstimateLDA
from python.params import params as p
import numpy as np

# Read in output file from PreprocessingSentences.py
complete_for_lda = pandas.read_csv(path_data + 'csv/complete_for_lda_{}_l.csv'.format(p['POStag']),
                                   sep='\t', na_filter=False)

np.random.seed(1) # setting random seed to get the same results each time

for no_b in np.arange(5, 106, 10):
    for no_a in np.arange(.2, .6, .05):
        # print(no_b, no_a)
        results_lda_std = LDACalibration(
                                type='tfidf',
                                dataframecolumn=complete_for_lda[complete_for_lda['Par_unique'] == 1]['paragraphs_{}_for_lda'.format(p['POStag'])],
                                # dataframecolumn=complete_for_lda['sentences_{}_for_lda'.format(p['POStag'])],
                                topics_start=4,
                                topics_limit=13,
                                topics_step=1,
                                topn=25,
                                num_words=30,
                                display_num_words=20,
                                metric=['coherence', 'hellinger', 'perplexity'],
                                no_below=no_b,
                                no_above=no_a,
                                alpha='auto',
                                eta='auto',
                                eval_every=5,
                                iterations=300,

                                random_state=123,
                                verbose=False,
                                display_plot=True,
                                save_plot=True,
                                save_model=True)

        print('############\n_no_below: {}, no_above: {},\n'.format(no_b, no_a), results_lda_std)


# _metric = 'perplexity'

# ldamodel, docsforlda, ldadict, ldacorpus \
#         = EstimateLDA(complete_for_lda[complete_for_lda['Art_unique'] == 1]['articles_{}_for_lda'.format(p['POStag'])],
#                       type='tfidf',
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

# complete_for_lda_calibr_tfidf = LDACalibration(
#                             type='tfidf',
#                             dataframecolumn=complete_for_lda[complete_for_lda['Art_unique'] == 1]['articles_{}_for_lda'.format(p['POStag'])],
#                             topics_start=1,
#                             topics_limit=20,
#                             topics_step=2,
#                             topn=200,
#                             num_words=200,
#                             metric=_metric,
#                             no_below=75,
#                             no_above=0.5,
#                             alpha='auto',
#                             eta='auto',
#                             eval_every=3,
#                             iterations=300,

#                             random_state=123,
#                             verbose=False,
#                             display_plot=True,
#                             save_plot=True)
