import pandas, time
from python.ConfigUser import path_processedarticles
from python._AnalysisFunctions import LDACalibration
from python.params import params as p

# Read in output file from PreprocessingSentences.py
complete_for_lda = pandas.read_csv(path_processedarticles + 'csv/complete_for_lda_{}_l.csv'.format(p['POStag_type']), sep='\t')

complete_for_lda_calibr = LDACalibration(
                            dataframecolumn=complete_for_lda['sentences_{}_for_lda'.format(p['POStag_type'])],
                            topics_start=1,
                            topics_limit=30,
                            topics_step=5,
                            topn=25,
                            num_words=25,
                            metric='hellinger',
                            no_below=0.1,
                            no_above=0.8,
                            alpha='auto',
                            eta='auto',
                            eval_every=10,
                            iterations=50,

                            random_state=123,
                            verbose=False,
                            display_plot=True)

