"""
Run preprocessing from here, adjust configurations for lda, for calibration
"""

import numpy as np

params = {

    # Specify POStag type
    'POStag_type':  'NNV',

    # Specify on which level to fit lda
    'lda_level_fit':        ['article'],
    # 'lda_level_fit':      ['sentence', 'paragraph', 'article'],

    # Specify of which level get dominant topics
    'lda_level_domtopic':     ['article'],
    # 'lda_level_domtopic': ['sentence', 'paragraph', 'article'],

    # EstimateLDA() parameters (Note: below parameters are passed to LdaModel())
    'type':         'standard',
    'no_below':     75, # filter extremes (words occurring in less than 20 docs, or in more than 50% of the docs)
    'no_above':     .5,
    'num_topics':   8,
    'alpha':        'auto',
    'eta':          'auto',
    'eval_every':   5,
    'iterations':   250,
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

