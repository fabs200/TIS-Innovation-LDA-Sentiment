"""
Specify parameters here, adjust configurations for lda, for its calibration and for adjusting lda-sentiment-plots
"""

import numpy as np

np.random.seed(1) # setting random seed to get the same results each time

params = {

    ### Sentiment params

    # select sentiment type ['sepldefault', 'seplmodified', 'sentiwsdefault', 'sentifinal']
    'sentiment_list': 'sentifinal',

    # drop short articles (int or None)
    'drop_article_lenght': None, #=2.5% percentile, We don't use this filter as we are also not filtering before lda

    # drop short sentences (int or None)
    'drop_sentence_lenght': None, #=2.5% percentile, We don't use this filter as we are also not filtering before lda

    # drop articles with low probability of assigned dominant topic (decimal number between 0 and 1 or None)
    'drop_prob_below': None, #=2.5% percentile, hardly any difference

    #drop sentences with (relatively) neutral sentiment score (either =0 or in range(-.1, .1), or set None
    'drop_senti_below': None, # hardly any difference
    'drop_senti_above': None, # hardly any difference


    ### LDA pramas
    
    # Specify POStag type
    'POStag':  'NNV',

    # Specify on which level to fit lda
    'lda_level_fit':        ['article'], # article, paragraph, sentence

    # Specify of which level get dominant topics
    'lda_level_domtopic':     ['article', 'paragraph', 'sentence'], # article, paragraph, sentence

    # EstimateLDA() parameters (Note: below parameters are passed to LdaModel())
    'type':         'tfidf',
    'no_below':     55, # Keep tokens which are contained in at least no_below documents
    'no_above':     .2, # Keep tokens which are contained in no more than no_above documents (fraction of total corpus)
    'num_topics':   5,
    'num_words':    50,
    'alpha':        'auto',
    'eta':          'auto',
    'eval_every':   5, # Log perplexity estimated every that many updates, =1 slows down training by ~2x
    'iterations':   300,
    'random_state': 123,

    # further parameters passed to LdaModel()
    'distributed':  False,
    'chunksize':    2000,
    'passes':       1,
    'update_every': 1,
    'decay':        0.5,
    'offset':       1.0,
    'gamma_threshold':0.001,
    'minimum_probability':0.01,
    'ns_conf':      None,
    'minimum_phi_value':0.01,
    'per_word_topics':False,
    'callbacks':    None,
    'dtype':        np.float32,
    'verbose':      True,

    # Preprocessing parameters
    'minwordinsent':2,
    'minwordlength':3

}



# current model (don't touch this)
currmodel = '{}_{}_{}_{}_k{}'.format(params['type'],
                                     params['POStag'],
                                     str(round(params['no_above'], ndigits=2)),
                                     str(round(params['no_below'], ndigits=3)),
                                     str(round(params['num_topics'], ndigits=0)))

# update params by currmodel
params['currmodel'] = currmodel

