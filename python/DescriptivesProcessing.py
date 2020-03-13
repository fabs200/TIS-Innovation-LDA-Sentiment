import os
from python.ProcessingFunctions import FlattenList, ListToFreqDict, ExportFreqDict
from python.PreprocessingArticles import noun_lemma_list

# Run whole script preprocessing_debug.py
# os.system("python {}/PreprocessingArticles.py".format(os.getcwd()))

# flatten list
flatnounlist = FlattenList(noun_lemma_list)

# make frequency lists
flatnounlist_freq = ListToFreqDict(flatnounlist)

# export dictionary
ExportFreqDict(flatnounlist_freq, filename='freqlist_nouns.xlsx')

