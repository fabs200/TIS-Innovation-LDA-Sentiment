"""
Specify parameters here, adjust configurations for lda and for calibration
"""

import numpy as np

np.random.seed(1) # setting random seed to get the same results each time

params = {

    # Specify POStag type
    'POStag':  'NNV',

    # Specify on which level to fit lda
    'lda_level_fit':        ['articles'], # article, paragraph, sentence

    # Specify of which level get dominant topics
    'lda_level_domtopic':     ['article'], # article, paragraph, sentence

    # EstimateLDA() parameters (Note: below parameters are passed to LdaModel())
    'type':         'tfidf',
    'no_below':     55, # Keep tokens which are contained in at least no_below documents
    'no_above':     .02, # Keep tokens which are contained in no more than no_above documents (fraction of total corpus)
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

